"""Idempotent, guarded patch for r2dreamer/dreamer.py.
Add a weighted positive-reward NLL without changing reward labels, actor or RSSM.
Run from the repository root: python install_reward_weight.py
"""
from pathlib import Path
import ast
import shutil
from datetime import datetime

path = Path('dreamer.py')
if not path.is_file():
    raise SystemExit('ERROR: run this script from the r2dreamer repository root (dreamer.py not found).')
source = path.read_text(encoding='utf-8')
if 'reward_pos_weight' in source:
    print('[SKIP] reward_pos_weight already present; no changes made.')
    raise SystemExit(0)

needle_init = '        self.act_entropy = float(config.act_entropy)'
needle_loss = '        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))'
if source.count(needle_init) != 1 or source.count(needle_loss) != 1:
    raise SystemExit('ERROR: repo code differs from verified version; no changes made. Ask for a tailored patch.')

source = source.replace(needle_init, needle_init + '\n        self.reward_pos_weight = float(getattr(config, "reward_pos_weight", 1.0))\n        if self.reward_pos_weight < 1.0:\n            raise ValueError("model.reward_pos_weight must be >= 1.0")', 1)

replacement = '''        # Weighted reward NLL: preserve scalar reward values and the original reward distribution.
        # A weight of 1.0 reproduces the unweighted original objective.
        reward_targets = to_f32(data["reward"])
        reward_nll = -self.reward(feat).log_prob(reward_targets)
        # Align reward [B,T,1] and log_prob [B,T] or [B,T,1] safely.
        reward_nll = reward_nll.reshape(B, T, -1).mean(dim=-1)
        reward_positive = (reward_targets.reshape(B, T, -1) > 0).any(dim=-1)
        reward_weights = torch.where(
            reward_positive,
            torch.full_like(reward_nll, self.reward_pos_weight),
            torch.ones_like(reward_nll),
        )
        # Normalized weights avoid changing the overall scale when positives are more common.
        losses["rew"] = (reward_weights * reward_nll).sum() / reward_weights.sum().clamp_min(1.0)
        metrics["reward/positive_fraction"] = reward_positive.float().mean().detach()
        metrics["reward/nll_unweighted"] = reward_nll.mean().detach()
        metrics["reward/nll_weighted"] = losses["rew"].detach()
        metrics["reward/pos_weight"] = torch.tensor(self.reward_pos_weight, device=reward_nll.device)
'''.rstrip('\n')
source = source.replace(needle_loss, replacement, 1)
ast.parse(source, filename='dreamer.py')
backup = path.with_name('dreamer.py.backup_before_reward_weight_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
shutil.copy2(path, backup)
path.write_text(source, encoding='utf-8')
print(f'[OK] Patched {path}; backup: {backup}')
print('[OK] Original behavior: model.reward_pos_weight=1.0 (or omitted)')
print('[OK] Weighted experiment: +model.reward_pos_weight=3.0')
