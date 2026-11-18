"""Compare two Dreamer reward models on EXACTLY the same Crafter transitions.

Place in r2dreamer root, run:
 python compare_reward_shared.py --episodes 30

Each episode uses the control actor (even episodes) or weighted actor (odd
episodes) to collect transitions; BOTH agents predict every transition.
No training and no checkpoint modifications.
"""
import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.amp import autocast

from dreamer import Dreamer
from envs import make_env


def obs_tensor(obs, is_first, device):
    return {
        'image': torch.as_tensor(np.asarray(obs['image']), device=device).unsqueeze(0),
        'is_first': torch.tensor([is_first], dtype=torch.bool, device=device),
    }


def model_reward(agent, state):
    return float(agent.reward(agent.rssm.get_feat(*state)).mode().float().reshape(-1)[0].item())


def stats(rows, prefix, threshold):
    y = np.asarray([x['reward_true'] for x in rows])
    out = {}
    for kind in ('prior', 'posterior'):
        p = np.asarray([x[f'{prefix}_{kind}'] for x in rows])
        positive, zero = y > 0, y == 0
        out[kind] = {
            'MAE': float(np.abs(p-y).mean()),
            'RMSE': float(np.sqrt(np.square(p-y).mean())),
            'positive_recall': float((p[positive] >= threshold).mean()) if positive.any() else None,
            'zero_reward_FPR': float((p[zero] >= threshold).mean()) if zero.any() else None,
            'mean_pred_on_positive': float(p[positive].mean()) if positive.any() else None,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, default=Path('logdir/crafter_reward_control_25k'))
    parser.add_argument('--weighted', type=Path, default=Path('logdir/crafter_reward_weighted_25k'))
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20261119)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--output', type=Path, default=Path('logdir/shared_reward_comparison'))
    args = parser.parse_args()
    if args.episodes < 2 or args.max_steps < 1:
        parser.error('Use at least 2 episodes and max-steps >= 1')

    dirs = {'control': args.control, 'weighted': args.weighted}
    paths = {k: (v / 'latest.pt').resolve() for k, v in dirs.items()}
    config_paths = {k: (v / '.hydra' / 'config.yaml').resolve() for k, v in dirs.items()}
    for p in list(paths.values()) + list(config_paths.values()):
        if not p.is_file():
            raise FileNotFoundError(p)
    configs = {k: OmegaConf.load(v) for k, v in config_paths.items()}
    for cfg in configs.values():
        cfg.model.compile = False
        cfg.env.seed = args.seed
    device = torch.device(str(configs['control'].device))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = make_env(configs['control'].env, 0)
    models = {}
    try:
        for name in dirs:
            model = Dreamer(copy.deepcopy(configs[name].model), env.observation_space, env.action_space).to(device)
            ckpt = torch.load(paths[name], map_location=device, weights_only=False)
            model.load_state_dict(ckpt['agent_state_dict'], strict=True)
            model.eval()
            models[name] = model
            print(f'Loaded {name}: {paths[name]}', flush=True)

        import crafter
        actions = list(crafter.constants.actions)
        records = []
        with torch.inference_mode():
            for ep in range(args.episodes):
                source = 'control' if ep % 2 == 0 else 'weighted'
                obs = env.reset()
                states = {}
                with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
                    for name, model in models.items():
                        st, dt = model.rssm.initial(1)
                        a0 = torch.zeros((1, model.act_dim), device=device)
                        embed = model.encoder(model.preprocess(obs_tensor(obs, True, device)))
                        s = model.rssm.obs_step(st, dt, a0, embed, torch.ones(1, device=device, dtype=torch.bool))
                        states[name] = s[:2]
                total_reward = 0.0
                for step in range(1, args.max_steps + 1):
                    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
                        actor = models[source]
                        action = actor.actor(actor.rssm.get_feat(*states[source])).rsample()
                        idx = int(action.argmax(-1).item())
                        predictions = {}
                        next_prior_states = {}
                        for name, model in models.items():
                            prior = model.rssm.img_step(*states[name], action)
                            next_prior_states[name] = prior
                            predictions[f'{name}_prior'] = model_reward(model, prior)

                    next_obs, reward, done, _ = env.step(action[0].float().cpu().numpy())
                    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
                        for name, model in models.items():
                            embed = model.encoder(model.preprocess(obs_tensor(next_obs, False, device)))
                            post = model.rssm.obs_step(*states[name], action, embed, torch.zeros(1,device=device,dtype=torch.bool))
                            states[name] = post[:2]
                            predictions[f'{name}_posterior'] = model_reward(model, post[:2])
                    records.append({
                        'episode': ep+1, 'source_policy':source, 'step':step,
                        'action_id':idx, 'action':actions[idx],
                        'reward_true':float(reward), **predictions,
                    })
                    total_reward += float(reward)
                    if done:
                        break
                print(f'Episode {ep+1}/{args.episodes} ({source}): steps={step}, reward={total_reward:+.2f}', flush=True)
    finally:
        try:env.close()
        except Exception:pass

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out/'shared_transitions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    y = np.array([x['reward_true'] for x in records])
    result = {
        'explanation':'Both models evaluated on identical transitions, half collected by each actor. Single-step teacher-forced prior; not a full open-loop rollout.',
        'episodes':args.episodes, 'seed':args.seed, 'transitions':len(records),
        'positive_rewards':int((y>0).sum()), 'zero_rewards':int((y==0).sum()),
        'threshold':args.threshold,
        'results':{k:stats(records,k,args.threshold) for k in models},
        'by_source':{source:{k:stats([r for r in records if r['source_policy']==source],k,args.threshold) for k in models} for source in models},
        'do_action':{k:stats([r for r in records if r['action']=='do'],k,args.threshold) for k in models}
    }
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n=== SHARED REWARD EVALUATION ===')
    for kind in ('prior','posterior'):
        for name in models:
            v=result['results'][name][kind]
            print(f'{kind:9} {name:9} MAE={v["MAE"]:.5f} RMSE={v["RMSE"]:.5f} Recall={v["positive_recall"]} FPR={v["zero_reward_FPR"]}')
    print(f'Output: {out.resolve()}')

if __name__=='__main__':
    main()
