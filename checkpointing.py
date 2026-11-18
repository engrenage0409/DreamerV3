"""Resumable r2dreamer checkpoint: weights, optimizer, replay and trainer counters.

WARNING: only load checkpoints you created/trust. torch.load(weights_only=False)
loads Python objects and must not be used on untrusted .pt files.
"""
from pathlib import Path
import os
import random
import shutil

import numpy as np
import torch
import tools


def save_checkpoint(logdir, agent, replay_buffer, trainer, step, update_count, numbered=True):
    directory = Path(logdir)
    directory.mkdir(parents=True, exist_ok=True)
    step = int(step)
    # Include original key names so evaluate_hud.py continues to load latest.pt.
    state = {
        "format_version": 1,
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        "replay_state_dict": replay_buffer._buffer.state_dict(),
        "trainer_state_dict": {
            "step": step,
            "update_count": int(update_count),
            "updates_last": trainer._updates_needed._last,
            "eval_last": trainer._should_eval._last,
            "log_last": trainer._should_log._last,
            "pretrain_once": trainer._should_pretrain._once,
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = directory / "latest.pt.tmp"
    dst = directory / "latest.pt"
    try:
        torch.save(state, tmp)
        os.replace(tmp, dst)  # Windows atomic replacement when same filesystem
        if numbered:
            shutil.copyfile(dst, directory / f"checkpoint_{step}.pt")
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"[CHECKPOINT] step={step} saved to {dst}", flush=True)


def load_checkpoint(path, agent, replay_buffer, restore_rng=False):
    path = Path(path)
    # weights_only=False is required for TorchRL replay and optimizer state.
    # ONLY load files that you created yourself / trust.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "replay_state_dict" not in state or "trainer_state_dict" not in state:
        raise ValueError(
            f"{path} is a model-only checkpoint (no Replay Buffer / trainer state). "
            "A fully resumable run must start with this patch installed. "
            "The earlier 5k latest.pt can still be evaluated with evaluate_hud.py."
        )
    agent.load_state_dict(state["agent_state_dict"], strict=True)
    tools.recursively_load_optim_state_dict(agent, state["optims_state_dict"])
    replay_buffer._buffer.load_state_dict(state["replay_state_dict"])
    # The small online queue only accelerates new windows; restarting it empty is safe.
    if restore_rng and "rng_state" in state:
        r = state["rng_state"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"].cpu())
        if torch.cuda.is_available() and r.get("cuda") is not None:
            torch.cuda.set_rng_state_all(r["cuda"])
    print(f"[RESUME] loaded {path} at environment step {state['trainer_state_dict']['step']}", flush=True)
    return state["trainer_state_dict"]
