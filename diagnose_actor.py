"""Compare Crafter DreamerV3 greedy (mode) and stochastic (sample) policies.

Usage in r2dreamer root (after installing the existing HUD patch):
  python diagnose_actor.py --run logdir/crafter_20k --episodes 10 --record-first

No training or checkpoint mutation. Results are written under RUN/actor_diagnostics/.
"""
import argparse
import copy
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch
from omegaconf import OmegaConf

from dreamer import Dreamer
from envs import make_envs


def close_envs(envs):
    if envs is None:
        return
    for worker in envs.envs:
        try:
            worker.close()
        except (OSError, EOFError):
            pass


def make_recorder(output_dir, policy, episode):
    from hud_recorder import HudRecorder

    return HudRecorder(output_dir / f"{policy}_episode_{episode:02d}.mp4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("logdir/crafter_20k"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record-first", action="store_true", help="Save first episode of EACH policy with HD HUD MP4")
    parser.add_argument("--max-steps", type=int, default=1000, help="Safety cap for each episode")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive")

    run = args.run.resolve()
    config_path = run / ".hydra" / "config.yaml"
    checkpoint_path = args.checkpoint or (run / "latest.pt")
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = OmegaConf.load(config_path)
    cfg.model.compile = False
    # Only one evaluation environment is necessary. Keep its seed identical
    # between policy modes to obtain paired terrain/episode sequences.
    cfg.env.eval_episode_num = 1
    cfg.trainer.eval_episode_num = 1
    output_dir = run / "actor_diagnostics"
    output_dir.mkdir(exist_ok=True)

    import crafter

    action_names = list(crafter.constants.actions)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Device: {cfg.device} | Crafter actions: {len(action_names)}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    if "agent_state_dict" not in checkpoint:
        raise KeyError("Checkpoint lacks 'agent_state_dict'")

    episode_rows = []
    per_mode_counts = {}
    try:
        for mode in ("mode", "sample"):
            # Create a NEW set of environments for each policy so both start
            # from the same Crafter episode seed sequence.
            train_envs = eval_envs = None
            try:
                train_envs, eval_envs, obs_space, act_space = make_envs(cfg.env)
                agent = Dreamer(copy.deepcopy(cfg.model), obs_space, act_space).to(cfg.device)
                agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
                agent.eval()
                counts = Counter()

                for ep in range(1, args.episodes + 1):
                    state = agent.get_initial_state(1)
                    action = state["prev_action"].clone()
                    done = torch.ones(1, device=agent.device, dtype=torch.bool)
                    total_reward = 0.0
                    steps = 0
                    achievements = {}
                    episode_actions = Counter()
                    recorder = make_recorder(output_dir, mode, ep) if args.record_first and ep == 1 else None
                    try:
                        while steps < args.max_steps:
                            was_reset = bool(done[0].item())
                            # Action belongs to the previous observation; the
                            # initial reset is NOT an environment action.
                            action_idx = None if was_reset else int(action[0].argmax().item())
                            trans, next_done = eval_envs.step(action.detach(), done)
                            trans = trans.to(agent.device, non_blocking=True)
                            reward = float(trans["reward"][0, 0].item())

                            if not was_reset:
                                episode_actions[action_idx] += 1
                                counts[action_idx] += 1
                                total_reward += reward
                                steps += 1
                                # Crafter's log_achievement values are
                                # cumulative within an episode, NOT per-frame
                                # events. Keep the last/max value, never sum.
                                for key, value in trans.items():
                                    if key.startswith("log_"):
                                        achieved = float(value[0, 0].item())
                                        name = key[4:]
                                        achievements[name] = max(achievements.get(name, 0.0), achieved)

                            if recorder is not None:
                                # HD render is requested only for the TWO
                                # recorded episodes, never during batch runs.
                                frame, inventory, native_ach, names = eval_envs.envs[0].video_snapshot()()
                                name = "START" if was_reset else names[action_idx]
                                recorder.add(frame, inventory, native_ach, action=name, reward=reward,
                                             step=steps)

                            if bool(next_done[0].item()):
                                break

                            with torch.no_grad():
                                action, state = agent.act(trans, state, eval=(mode == "mode"))
                            done = next_done.to(agent.device)
                    finally:
                        if recorder is not None:
                            recorder.close()

                    events = {k: int(v) for k, v in achievements.items() if v > 0}
                    row = {
                        "policy": mode,
                        "episode": ep,
                        "reward": round(total_reward, 6),
                        "length": steps,
                        "achievements_distinct": len(events),
                        "achievements": json.dumps(events, ensure_ascii=False, sort_keys=True),
                        "top_action": action_names[episode_actions.most_common(1)[0][0]] if episode_actions else "NONE",
                        "top_action_fraction": round(episode_actions.most_common(1)[0][1] / steps, 6) if steps else 0.0,
                        "time_limit_reached": steps >= args.max_steps,
                    }
                    episode_rows.append(row)
                    print(f"[{mode:6}] episode {ep:02d}/{args.episodes}: "
                          f"reward={total_reward:+.3f}, length={steps}, "
                          f"achievements={len(events)}, top={row['top_action']} "
                          f"({100*row['top_action_fraction']:.1f}%)", flush=True)
                per_mode_counts[mode] = counts
            finally:
                close_envs(eval_envs)
                close_envs(train_envs)

        with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0]))
            writer.writeheader()
            writer.writerows(episode_rows)

        freq_rows = []
        summary = {
            "checkpoint": str(checkpoint_path),
            "run": str(run),
            "episode_count_each": args.episodes,
            "evaluation": "mode uses argmax distribution action; sample draws stochastic action",
            "note": "Crafter achievement values are cumulative; counted as final/max for each episode, not summed over frames.",
            "policies": {},
        }
        for mode in ("mode", "sample"):
            rows = [r for r in episode_rows if r["policy"] == mode]
            counts = per_mode_counts[mode]
            total = sum(counts.values())
            probs = [counts.get(i, 0) / total if total else 0.0 for i in range(len(action_names))]
            empirical_entropy = -sum(p * math.log(p) for p in probs if p > 0)
            for i, name in enumerate(action_names):
                freq_rows.append({"policy": mode, "action_id": i, "action": name,
                                  "count": counts.get(i, 0), "frequency": round(probs[i], 8)})
            summary["policies"][mode] = {
                "mean_reward": mean(r["reward"] for r in rows),
                "std_reward": pstdev(r["reward"] for r in rows),
                "mean_length": mean(r["length"] for r in rows),
                "mean_distinct_achievements": mean(r["achievements_distinct"] for r in rows),
                "empirical_action_entropy": empirical_entropy,
                "top_actions": [
                    {"name": action_names[i], "count": count, "fraction": count / total}
                    for i, count in counts.most_common(5)
                ],
            }

        with (output_dir / "action_frequency.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(freq_rows[0]))
            writer.writeheader()
            writer.writerows(freq_rows)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("\n--- Mode versus sample ---")
        for mode, data in summary["policies"].items():
            print(f"{mode:6} | mean reward {data['mean_reward']:+.3f} "
                  f"(std {data['std_reward']:.3f}) | "
                  f"mean length {data['mean_length']:.1f} | "
                  f"action entropy (empirical) {data['empirical_action_entropy']:.3f}")
            print("       top actions:", ", ".join(
                f"{x['name']}={100*x['fraction']:.1f}%" for x in data["top_actions"]
            ))
        print("\nSaved:", output_dir / "summary.json")
        print("       ", output_dir / "episodes.csv")
        print("       ", output_dir / "action_frequency.csv")
    except BaseException:
        print("Diagnostic failed before writing the final report.", flush=True)
        raise


if __name__ == "__main__":
    main()
