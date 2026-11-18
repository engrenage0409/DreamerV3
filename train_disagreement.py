"""Train/validate lightweight disagreement heads on real Crafter transitions.

Stage 5.3a ONLY. Frozen Dreamer encoder/RSSM/actor; no intrinsic bonus yet.

python train_disagreement.py --run logdir/crafter_reward_weighted_25k \
    --episodes 30 --validation-episodes 8 --epochs 20 --batch-size 128
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.amp import autocast

from disagreement_ensemble import DisagreementEnsemble
from dreamer import Dreamer
from envs import make_env


def _encode_observation(agent, observation, device):
    # Crafter image is [H,W,C]. Encoder expects [B,H,W,C].
    image = torch.as_tensor(np.asarray(observation["image"]), device=device).unsqueeze(0)
    return agent.encoder(agent.preprocess({"image": image}))


@torch.no_grad()
def collect_episodes(agent, env, device, episodes, seed, policy, max_steps):
    torch.manual_seed(seed)
    np.random.seed(seed)
    samples = []
    for ep in range(episodes):
        obs = env.reset()
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            stoch, deter = agent.rssm.initial(1)
            embedding = _encode_observation(agent, obs, device)
            zero_action = torch.zeros((1, agent.act_dim), device=device)
            stoch, deter, _ = agent.rssm.obs_step(
                stoch, deter, zero_action, embedding,
                torch.ones((1,), dtype=torch.bool, device=device),
            )
        reward_sum = 0.0
        length = 0
        for _ in range(max_steps):
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                feature = agent.rssm.get_feat(stoch, deter)
                if policy == "random":
                    action_index = torch.randint(agent.act_dim, (1,), device=device)
                    action = torch.nn.functional.one_hot(action_index, agent.act_dim).float()
                else:
                    distribution = agent.actor(feature)
                    action = (distribution.mode if policy == "mode" else distribution.rsample()).float()
                next_obs, reward, done, _ = env.step(action[0].cpu().numpy())
                next_embedding = _encode_observation(agent, next_obs, device)
                next_stoch, next_deter, _ = agent.rssm.obs_step(
                    stoch, deter, action, next_embedding,
                    torch.zeros((1,), dtype=torch.bool, device=device),
                )
                next_feature = agent.rssm.get_feat(next_stoch, next_deter)
            # Use float32 CPU arrays for tiny 3-head model: no large video/replay buffers here.
            samples.append((feature[0].float().cpu(),
                            action[0].float().cpu(),
                            next_feature[0].float().cpu()))
            stoch, deter = next_stoch, next_deter
            length += 1
            reward_sum += float(reward)
            if done:
                break
        print(f"[COLLECT {ep+1}/{episodes}] steps={length} reward={reward_sum:.2f}", flush=True)
    if not samples:
        raise RuntimeError("No transitions collected")
    return tuple(torch.stack([row[k] for row in samples]) for k in range(3))


@torch.no_grad()
def evaluate(ensemble, dataset, device, batch_size=512):
    ensemble.eval()
    total_mse = total_disagreement = 0.0
    n = dataset[0].shape[0]
    for start in range(0, n, batch_size):
        current, action, nxt = [part[start:start + batch_size].to(device) for part in dataset]
        predictions = ensemble(current, action)
        target = ensemble.project_target(nxt)
        loss = (predictions.float() - target.unsqueeze(0)).square().mean()
        disagreement = ensemble.disagreement(current, action).mean()
        size = current.shape[0]
        total_mse += float(loss) * size
        total_disagreement += float(disagreement) * size
    return {"mse": total_mse / n, "mean_disagreement": total_disagreement / n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("logdir/crafter_reward_weighted_25k"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--policy", choices=["sample", "mode", "random"], default="sample")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()
    if min(args.episodes, args.validation_episodes, args.epochs, args.batch_size, args.max_steps) < 1:
        parser.error("Episode counts, epochs, batch size, max steps must be positive")

    run = args.run.resolve()
    config_file, checkpoint_file = run / ".hydra/config.yaml", run / "latest.pt"
    if not config_file.is_file() or not checkpoint_file.is_file():
        raise FileNotFoundError(f"Missing run config/checkpoint: {config_file} / {checkpoint_file}")
    cfg = OmegaConf.load(config_file)
    cfg.model.compile = False
    cfg.env.seed = args.seed
    device = torch.device(str(cfg.device))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(cfg.env, 0)
    try:
        agent = Dreamer(copy.deepcopy(cfg.model), env.observation_space, env.action_space).to(device)
        # WARNING: only load trusted checkpoints created by your own training.
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
        agent.eval()
        for parameter in agent.parameters():
            parameter.requires_grad_(False)
        training = collect_episodes(agent, env, device, args.episodes,
                                    args.seed, args.policy, args.max_steps)
    finally:
        env.close()

    # A different Crafter RNG seed and distinct episodes for validation.
    cfg.env.seed = args.seed + 10000
    env = make_env(cfg.env, 0)
    try:
        validation = collect_episodes(agent, env, device, args.validation_episodes,
                                      args.seed + 10000, args.policy, args.max_steps)
    finally:
        env.close()

    ensemble = DisagreementEnsemble(
        feature_dim=agent.rssm.feat_size,
        action_dim=agent.act_dim,
        target_dim=args.target_dim,
        hidden_dim=args.hidden_dim,
        num_heads=3,
        seed=args.seed,
    ).to(device)
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=args.lr)
    baseline = evaluate(ensemble, validation, device)
    print("[BEFORE]", baseline, flush=True)
    shuffle = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    history = []

    for epoch in range(1, args.epochs + 1):
        ensemble.train()
        order = torch.randperm(training[0].shape[0], generator=shuffle)
        loss_sum = 0.0
        count = 0
        for ids in order.split(args.batch_size):
            current, action, nxt = [part[ids].to(device) for part in training]
            loss, _ = ensemble.bootstrap_loss(current, action, nxt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), 10.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            count += len(ids)
        validated = evaluate(ensemble, validation, device)
        row = {"epoch": epoch, "train_loss": loss_sum / count, **validated}
        history.append(row)
        print(f"[EPOCH {epoch:02d}] train={row['train_loss']:.5f} "
              f"val_mse={row['mse']:.5f} disagreement={row['mean_disagreement']:.5f}",
              flush=True)

    output = run / "ensemble_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "ensemble_state_dict": ensemble.cpu().state_dict(),
        "feature_dim": agent.rssm.feat_size,
        "action_dim": agent.act_dim,
        "target_dim": args.target_dim,
        "hidden_dim": args.hidden_dim,
        "num_heads": 3,
        "seed": args.seed,
        "source_checkpoint": str(checkpoint_file),
    }, output / "disagreement_ensemble.pt")
    summary = {
        "phase": "5.3a",
        "train_transitions": training[0].shape[0],
        "validation_transitions": validation[0].shape[0],
        "train_episodes": args.episodes,
        "validation_episodes": args.validation_episodes,
        "before": baseline,
        "after": history[-1],
        "history": history,
        "note": "Holdout one-step posterior feature prediction; no actor intrinsic reward."
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Results in {output}", flush=True)


if __name__ == "__main__":
    main()
