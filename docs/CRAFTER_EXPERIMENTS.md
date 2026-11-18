# Crafter DreamerV3 experiments (Windows / RTX 5060 8GB)

This experimental branch extends the upstream R2-Dreamer codebase with a DreamerV3-style Crafter training and evaluation workflow. Keep upstream attribution and project license intact. Experiment values below are from local runs, not published benchmarks or a reproduced Crafter Score.

## Implemented modules

| Module | Files | Purpose |
|---|---|---|
| Replay and checkpoint continuation | `checkpointing.py`, `train.py`, `trainer.py` | Resume optimizer, replay and step counters from trusted local checkpoints |
| Exploration | `trainer.py`, `install_epsilon_exploration.py` | Epsilon-greedy random action collection (not eval) |
| Reward-weight ablation | `dreamer.py`, `install_reward_weight.py` | Weighted **positive-target NLL** with switch `+model.reward_pos_weight=1.0/3.0` |
| Actor diagnostic | `diagnose_actor.py` | 10/30-episode mode/sample runs, action histogram, real achievements |
| Reward diagnostic | `diagnose_reward.py`, `compare_reward_shared.py` | Posterior/prior reward accuracy, plus identical-transition comparison |
| Windows visualization | `evaluate_hud.py`, `hud_recorder.py`, `tb_video_win.py` | HUD video and TensorBoard export support |

## Evaluation caveats

- Crafter `log_achievement` values are **cumulative within an episode**. Never sum them frame by frame. Use final/max per episode.
- Episode reward in these experiments is **not** the official Crafter Score (geometric mean of achievement success rates).
- Actual action histogram entropy is **not** necessarily policy entropy at each state. High action diversity can still be purposeless action cycling.
- Reward comparison `prior` is one-step and teacher-forced with RSSM states; it is **not** a 15-step imagination-rollout accuracy measurement.
- Different actors collect different transitions; shared-transition comparisons mitigate state-distribution confounding, but 30 evaluation episodes are not enough for strong significance claims.
- Negative, zero, and positive rewards all occur: always report recall and FPR thresholds explicitly.

## Reproduced experimental observations

| Checkpoint | Mode mean reward | Sample mean reward | Sample empirical entropy | Sample distinct achievements |
|---|---:|---:|---:|---:|
| Baseline 5k | 1.5 | 1.9 | 1.130 | 2.8 |
| Entropy 0.003 5k | 2.0 | 2.2 | 1.548 | 3.1 |
| Entropy + epsilon 0.05 5k | 1.9 | 2.6 | 1.891 | 3.5 |
| Entropy + epsilon 0.05 20k | 2.3 | 2.6 | 2.686 | 3.6 |
| Control reward weight 1.0 at 25k (30 episodes each) | 1.93 | 1.97 | 2.711 | 2.87 |
| Weighted reward weight 3.0 at 25k (30 episodes each) | 2.23 | 2.33 | 2.672 | 3.30 |

The 5k/20k actor diagnostics used 10 episodes per policy; the 25k pair used 30 each. Do not compare small differences without additional seeds.

### Shared-transition reward head test (5,209 transitions; 130 positive, threshold 0.05)

| Metric | Control | Weighted |
|---|---:|---:|
| Prior MAE | 0.03527 | 0.03613 |
| Prior RMSE | 0.14088 | 0.14677 |
| Prior positive recall | 36.15% | 36.92% |
| Prior FPR on zero rewards | 5.29% | 4.66% |
| Posterior positive recall | 59.23% | 63.08% |
| Posterior FPR on zero rewards | 4.08% | 3.77% |

Weighted policy scores were higher in this limited evaluation, but improving reward NLL did **not** reliably solve prior reward prediction. In particular, `do` actions retain many low-threshold false positives.

## Next: Phase 5.3, latent disagreement ensemble

**Stage A (safe / offline):** freeze the pretrained Encoder and RSSM, collect real Crafter transitions, train three independently initialized and bootstrapped small MLP heads to predict a fixed projection of the *next posterior* RSSM feature from (current posterior feature, actual action). Report held-out MSE and inter-head disagreement. This stage does not change actor rewards or the base checkpoint.

**Stage B (after Stage A validation):** carefully introduce a normalized, bounded disagreement-based intrinsic bonus into actor/value imagination targets; compare with a same-budget no-intrinsic baseline and watch for in-place action loops. An ensemble's variance can capture noise or model errors rather than useful novelty; it is not a guarantee of better behavior.

## Safety and repository hygiene

- Do **not** commit `logdir/`, trusted local checkpoint blobs (`*.pt`), replay buffers, credentials, or MP4 outputs.
- Load only locally created, trusted PyTorch checkpoints; resuming replay may use `torch.load(weights_only=False)`.
- Keep experiment branch separate from upstream/default branch until changes have been reviewed.
