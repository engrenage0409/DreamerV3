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


## Phase 5.3b: bounded intrinsic bonus in imagined trajectories

Implementation: `disagreement_ensemble.py`, `dreamer.py` (`configure_intrinsic` and `_cal_grad`), and `train.py`. The same 25k world model and replay checkpoint are used as the starting point for both runs. Ensemble weights are **frozen**, loaded only when `intrinsic_beta>0`, and never registered in Dreamer's `state_dict`; existing checkpoints load strictly. The bonus is applied only to **imagined** rewards for the Actor-Critic and not to replay labels or the Reward Head training loss.

The model computes the ensemble variance on `(imag_feat[t], imag_action[t])`. That variance is attached to `imag_reward[t+1]`, not the same-time reward; the first imagined timestep has no intrinsic bonus. With `beta=0.05`, `scale=0.01035`, and `clip=3.0`, the maximum intrinsic bonus per imagined step is **0.15**.

**Important limitations**:
- The ensemble is static while the resumed world model evolves, so feature distribution drift may increase disagreement for reasons unrelated to meaningful novelty. Limit initial continuation to 5k environment steps and measure it.
- The ensemble was trained on only about 5k transitions from the original actor; high disagreement can reflect model error or noise, not just epistemic uncertainty. A result must be judged by actual Crafter achievements and repeated-action behavior.
- `r_total` is used inside imagined actor/value targets. The replay value loss still uses real environment reward and an imagined-value bootstrap, so this is a prototype exploration ablation rather than a complete separate intrinsic critic.
- The two runs may diverge stochastically even from the same checkpoint; results from one seed are suggestive only.
- All experiment checkpoints are local; do not commit `*.pt` to Git.

### Windows commands (PowerShell)

Ensure the new GitHub branch code has been fetched and fast-forwarded, and that the 25k weighted checkpoint and `ensemble_diagnostics/disagreement_ensemble.pt` exist locally. To keep the ablation narrow, use exactly the same configuration in both runs except `intrinsic_beta`.

```powershell
# CONTROL: intrinsic beta zero
python train.py `
  env=crafter model.rep_loss=dreamer model.compile=false `
  model.log_grads=true model.act_entropy=0.003 `
  +model.reward_pos_weight=3.0 `
  batch_size=16 batch_length=64 `
  env.steps=30000 env.train_ratio=512 `
  env.eval_episode_num=10 trainer.eval_every=5000 `
  trainer.update_log_every=500 trainer.video_pred_log=false `
  buffer.max_size=30000 buffer.storage_device=cpu `
  device=cuda:0 seed=0 `
  +checkpoint_every=5000 +trainer.explore_epsilon=0.05 `
  +resume=logdir/crafter_reward_weighted_25k/checkpoint_25000.pt `
  +intrinsic_beta=0.0 `
  logdir=logdir/crafter_intrinsic_control_30k
```

```powershell
# TREATMENT: identical resume and settings; small frozen intrinsic bonus
python train.py `
  env=crafter model.rep_loss=dreamer model.compile=false `
  model.log_grads=true model.act_entropy=0.003 `
  +model.reward_pos_weight=3.0 `
  batch_size=16 batch_length=64 `
  env.steps=30000 env.train_ratio=512 `
  env.eval_episode_num=10 trainer.eval_every=5000 `
  trainer.update_log_every=500 trainer.video_pred_log=false `
  buffer.max_size=30000 buffer.storage_device=cpu `
  device=cuda:0 seed=0 `
  +checkpoint_every=5000 +trainer.explore_epsilon=0.05 `
  +resume=logdir/crafter_reward_weighted_25k/checkpoint_25000.pt `
  +intrinsic_beta=0.05 `
  +intrinsic_scale=0.01035 `
  +intrinsic_clip=3.0 `
  +intrinsic_ensemble_path=logdir/crafter_reward_weighted_25k/ensemble_diagnostics/disagreement_ensemble.pt `
  logdir=logdir/crafter_intrinsic_bonus_30k
```

Replace the resume filename with `latest.pt` if `checkpoint_25000.pt` is absent, **only after verifying it is the 25k checkpoint**. Inspect TensorBoard scalar names `train/intrinsic/disagreement_mean`, `train/intrinsic/bonus_mean`, `train/intrinsic/bonus_max`, `episode/eval_score`, `train/loss/policy`, and `train/loss/value`. The intrinsic logs appear only for treatment.

Then compare at least 30 independent `mode` and `sample` episodes per checkpoint with `diagnose_actor.py` (the saved eval episode count of 10 is an intermediate signal, not an adequate final comparison). Also check `do` action repetition and distinct achieved tasks before declaring exploration improved.
