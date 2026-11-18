"""Dreamer/Crafter reward model diagnosis with fresh, real environment transitions.

Run from r2dreamer repository root:
  python diagnose_reward.py --run logdir/crafter_epsilon_005_5k --episodes 10 --policy sample

This does NOT modify the checkpoint, replay buffer, or training files.
Posterior: R(s_{t+1}^{post}) after seeing the next image.
Prior:     R(s_{t+1}^{prior}) predicted from (s_t, a_t) before seeing the next image.
Both are evaluated against the true environment reward r_{t+1}.
"""
import argparse
import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.amp import autocast

from dreamer import Dreamer
from envs import make_env


def as_model_obs(image, is_first, device):
    # Dreamer encoder expects (batch=1, height, width, channels), uint8.
    return {
        'image': torch.as_tensor(np.asarray(image['image']), device=device).unsqueeze(0),
        'is_first': torch.tensor([is_first], device=device, dtype=torch.bool),
    }


def get_reward(agent, state):
    return float(agent.reward(agent.rssm.get_feat(*state)).mode().reshape(-1)[0].float().item())


def summarize(records, threshold):
    n = len(records)
    if not n:
        return {'count': 0}
    y = np.asarray([r['reward_true'] for r in records], dtype=np.float64)
    out = {'count': n, 'true_positive_reward_count': int((y > 0).sum()),
           'true_negative_reward_count': int((y < 0).sum()),
           'true_zero_reward_count': int((y == 0).sum())}
    for name in ('prior', 'posterior'):
        pred = np.asarray([r[f'reward_{name}'] for r in records], dtype=np.float64)
        positive = y > 0
        zero = y == 0
        predicted_positive = pred >= threshold
        out[name] = {
            'mae': float(np.abs(pred - y).mean()),
            'rmse': float(np.sqrt(np.mean((pred - y) ** 2))),
            'bias_mean_pred_minus_true': float(np.mean(pred - y)),
            'positive_reward_recall': float(predicted_positive[positive].mean()) if positive.any() else None,
            'false_positive_rate_on_zero_rewards': float(predicted_positive[zero].mean()) if zero.any() else None,
            'mean_prediction_when_reward_zero': float(pred[zero].mean()) if zero.any() else None,
            'mean_prediction_when_reward_positive': float(pred[positive].mean()) if positive.any() else None,
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('logdir/crafter_epsilon_005_5k'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--policy', choices=['sample', 'mode', 'random'], default='sample')
    p.add_argument('--threshold', type=float, default=0.5, help='Predicted reward >= threshold counts as a positive-reward prediction')
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261118, help='Use a separate evaluation seed (not training seed)')
    args = p.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        p.error('episodes and max-steps must be positive')
    run = args.run.expanduser().resolve()
    config_file = run / '.hydra' / 'config.yaml'
    checkpoint_file = (args.checkpoint or run / 'latest.pt').expanduser().resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f'Missing Hydra config: {config_file}')
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {checkpoint_file}')

    cfg = OmegaConf.load(config_file)
    cfg.model.compile = False
    cfg.env.seed = args.seed
    # Construct only one synchronous environment, avoiding 10 subprocesses/HD video.
    env = make_env(cfg.env, 0)
    device = torch.device(str(cfg.device))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    try:
        agent = Dreamer(copy.deepcopy(cfg.model), env.observation_space, env.action_space).to(device)
        ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt['agent_state_dict'], strict=True)
        agent.eval()

        import crafter
        action_names = list(crafter.constants.actions)
        records = []
        print(f'Checkpoint: {checkpoint_file}')
        print(f'Episodes: {args.episodes}, policy: {args.policy}, reward threshold: {args.threshold}')
        with torch.inference_mode():
            for ep in range(1, args.episodes + 1):
                obs = env.reset()
                stoch, deter = agent.rssm.initial(1)
                zero_action = torch.zeros((1, agent.act_dim), device=device, dtype=torch.float32)
                with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    initial_embed = agent.encoder(agent.preprocess(as_model_obs(obs, True, device)))
                    stoch, deter, _ = agent.rssm.obs_step(
                        stoch, deter, zero_action, initial_embed, torch.ones(1, device=device, dtype=torch.bool)
                    )
                ep_reward = 0.0
                for step in range(1, args.max_steps + 1):
                    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                        feat = agent.rssm.get_feat(stoch, deter)
                        policy = agent.actor(feat)
                        if args.policy == 'random':
                            idx = int(torch.randint(agent.act_dim, (1,), device=device).item())
                            action = torch.nn.functional.one_hot(
                                torch.tensor([idx], device=device), num_classes=agent.act_dim
                            ).float()
                        else:
                            action = policy.mode if args.policy == 'mode' else policy.rsample()
                            idx = int(action.argmax(dim=-1).item())
                        # PRIOR: predict next latent without next observation, then predict next reward.
                        next_prior = agent.rssm.img_step(stoch, deter, action)
                        reward_prior = get_reward(agent, next_prior)

                    # The real env gets the same action that was used for the RSSM prior.
                    next_obs, true_reward, done, _ = env.step(action[0].float().cpu().numpy())
                    # POSTERIOR: update with the true next observation.
                    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                        embed_next = agent.encoder(agent.preprocess(as_model_obs(next_obs, False, device)))
                        next_post = agent.rssm.obs_step(
                            stoch, deter, action, embed_next,
                            torch.zeros(1, device=device, dtype=torch.bool)
                        )
                        reward_post = get_reward(agent, next_post[:2])
                    records.append({
                        'episode': ep, 'step': step, 'action_id': idx,
                        'action': action_names[idx], 'reward_true': float(true_reward),
                        'reward_prior': reward_prior, 'reward_posterior': reward_post,
                        'abs_err_prior': abs(reward_prior - float(true_reward)),
                        'abs_err_posterior': abs(reward_post - float(true_reward)),
                        'done': int(bool(done)),
                    })
                    ep_reward += float(true_reward)
                    stoch, deter = next_post[:2]
                    if done:
                        break
                print(f'Episode {ep:02d}/{args.episodes}: {step} steps, reward {ep_reward:+.2f}', flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    outdir = run / 'reward_diagnostics'
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / 'transitions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    grouped = defaultdict(list)
    for row in records:
        grouped[row['action']].append(row)
    result = {
        'checkpoint': str(checkpoint_file), 'evaluation_seed': args.seed,
        'policy': args.policy, 'episodes': args.episodes,
        'threshold': args.threshold,
        'note': ('Fresh on-policy evaluation, not a replay-buffer sample. '
                 'Prior predicts reward from previous posterior state and chosen action before seeing next image. '
                 'Posterior uses next observed image. Predictions are conditioned on sampled RSSM states.'),
        'overall': summarize(records, args.threshold),
        'per_action': {name: summarize(rows, args.threshold) for name, rows in sorted(grouped.items())},
    }
    report = outdir / 'summary.json'
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n=== Reward diagnostics ===')
    for name in ('prior', 'posterior'):
        m = result['overall'][name]
        print(f"{name:9} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
              f"FPR@{args.threshold}={m['false_positive_rate_on_zero_rewards']} "
              f"Recall@{args.threshold}={m['positive_reward_recall']}")
    print(f'Positive-reward transitions: {result["overall"]["true_positive_reward_count"]}')
    print(f'Saved: {report}\n       {outdir / "transitions.csv"}')


if __name__ == '__main__':
    main()
