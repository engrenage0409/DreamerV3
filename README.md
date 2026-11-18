# DreamerV3 World Model Lab — Crafter Research Extension

> **Experimental research branch `dreamerv3-experiments`** — personal DreamerV3/Crafter training and ablations built on the upstream R2-Dreamer implementation. Original authorship, citation and upstream repository instructions are preserved below. This branch does **not** claim authorship of the R2-Dreamer paper or an official Crafter benchmark result.

**[📘 Full bilingual-context research report: CarRacing → Crafter, Action Bias, Reward and Ensemble ablations](docs/CRAFTER_EXPERIMENTS.md)**

The report reconstructs the development journey, including early **CarRacing World Model loss decreasing without a successful driving policy**, Crafter's early **`do` / `place_plant` / `move_down` action bias**, entropy regularization and ε-greedy improvements, Reward Head diagnostics on shared transitions, three-head Latent Disagreement prediction, and the **negative 25k→30k Intrinsic Reward ablation**.

### Main findings (local experiments, limited training budget)

| Experiment | Observation |
|---|---|
| Crafter 5k baseline | Mode / Sample game reward ~1.50 / 1.90; notable action preference |
| Entropy coefficient 0.0003→0.003 | Sample game reward ~1.90→2.20 at 5k, but mode still biased to `move_down` in one run |
| Entropy + ε-greedy collection | Helped diversify behavior; 20k sample reward ~2.60 |
| Weighted reward NLL at 25k | Weighted sample reward 2.33 vs control 1.97, **but** identical-transition Prior positive recall 36.92% vs 36.15% |
| Three-head latent predictor | Validation MSE 1.0025→0.2783 over 20 epochs; this does not establish useful novelty |
| **Control vs Intrinsic at 30k** | **Mode game reward 3.13 vs 2.20**; mode distinct achievements 4.10 vs 3.17; intrusive bonus was detrimental under this setup |

**Current best candidate:** `crafter_intrinsic_control_30k` (no intrinsic reward). The 30k intrinsic experiment is recorded as a negative ablation; higher prediction accuracy or reward bonuses do not guarantee improved policy learning. Values above are small-budget observations with different evaluation samples where noted, **not official Crafter Score**.

**Implementation map:** `train.py`, `trainer.py`, `dreamer.py`, `rssm.py`, `checkpointing.py`, `diagnose_actor.py`, `diagnose_reward.py`, `compare_reward_shared.py`, `train_disagreement.py`, `disagreement_ensemble.py`, `evaluate_hud.py`, `hud_recorder.py`. See the complete report for scripts, matched experiment commands, reward definitions and caveats.

---

## R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation

This repository provides the official implementation of [R2-Dreamer][r2dreamer] (ICLR 2026), a decoder-free, augmentation-free world model with redundancy-reducing regularization on latent representations. It also includes an efficient PyTorch DreamerV3 reproduction that trains **~5x faster** than a widely used [codebase][dreamerv3-torch], along with other baselines. Selecting R2-Dreamer via the config provides an additional **~1.6x speedup** over this baseline.

## Instructions

Install dependencies. This repository is tested with Ubuntu 24.04 and Python 3.11.

Installing via a virtual env like [uv](https://docs.astral.sh/uv/) is recommended. If you prefer Docker, follow [`docs/docker.md`](docs/docker.md).

```bash
# Select the environments you need, or use "all" to install everything.
# Available extras: dmc, atari, crafter, metaworld, memorymaze, isaaclab, all
pip install -e ".[dmc,metaworld]"
```

Run training on default settings:

```bash
python3 train.py logdir=./logdir/test
```

Monitoring results:
```bash
tensorboard --logdir ./logdir
```

Switching algorithms:

```bash
# Choose an algorithm via model.rep_loss:
# r2dreamer|dreamer|infonce|dreamerpro
python3 train.py model.rep_loss=r2dreamer
```

For easier code reading, inline tensor shape annotations are provided. See [`docs/tensor_shapes.md`](docs/tensor_shapes.md).


## Available Benchmarks
At the moment, the following benchmarks are available in this repository.

| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [Meta-World](https://github.com/Farama-Foundation/Metaworld) | Image | Continuous | 1M | Robotic manipulation with complex contact interactions.|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [DMC Subtle](envs/dmc_subtle.py) | Image | Continuous |1M| DeepMind Control Suite with tiny task-relevant objects. |
| [Atari 100k](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) | State / Image | Continuous | 0.5M–1M | GPU-parallelized robotic simulation. |

Use Hydra to select a benchmark and a specific task using `env` and `env.task`, respectively.

```bash
python3 train.py ... env=dmc_vision env.task=dmc_walker_walk
```

## Headless rendering

If you run MuJoCo-based environments (DMC / MetaWorld) on headless machines, you may need to set `MUJOCO_GL` for offscreen rendering. **Using EGL is recommended** as it accelerates rendering, leading to faster simulation throughput.

```bash
# For example, when using EGL (GPU)
export MUJOCO_GL=egl
# (optional) Choose which GPU EGL uses
export MUJOCO_EGL_DEVICE_ID=0
```

More details: [Working with MuJoCo-based environments](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)

## Code formatting

If you want automatic formatting/basic checks before commits, you can enable `pre-commit`:

```bash
pip install pre-commit
# This sets up a pre-commit hook so that checks are run every time you commit
pre-commit install
# Manual pre-commit run on all files
pre-commit run --all-files
```

## Troubleshooting

If you encounter installation or runtime issues, please refer to our [Troubleshooting Guide](docs/troubleshooting.md).

## Citation

If you find this code useful, please consider citing:

```bibtex
@inproceedings{
morihira2026rdreamer,
title={R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation},
author={Naoki Morihira and Amal Nahar and Kartik Bharadwaj and Yasuhiro Kato and Akinobu Hayashi and Tatsuya Harada},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Je2QqXrcQq}
}
```

[r2dreamer]: https://openreview.net/forum?id=Je2QqXrcQq&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)
[dreamerv3-torch]: https://github.com/NM512/dreamerv3-torch
