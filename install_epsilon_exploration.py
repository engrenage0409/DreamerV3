"""Install optional epsilon-random data collection in NM512/r2dreamer trainer.py.

Usage (from repo root): python install_epsilon_exploration.py
Then: +trainer.explore_epsilon=0.05

Only modifies online TRAIN collection. Eval, model architecture, reward, and checkpoint
format remain unchanged. Backs up original trainer.py before editing.
"""
from pathlib import Path
import shutil

p = Path('trainer.py')
if not p.is_file():
    raise SystemExit('Run from r2dreamer repo root (trainer.py not found).')
s = p.read_text(encoding='utf-8')
if 'EPSILON_EXPLORATION_PATCH_V1' in s:
    print('[OK] Epsilon exploration already installed.')
    raise SystemExit(0)

anchor_init = '        self._action_repeat = config.action_repeat\n'
new_init = '''        self._action_repeat = config.action_repeat
        # EPSILON_EXPLORATION_PATCH_V1: epsilon-random TRAIN collection only.
        self.explore_epsilon = float(getattr(config, "explore_epsilon", 0.0))
        if not (0.0 <= self.explore_epsilon <= 1.0):
            raise ValueError(f"explore_epsilon must be in [0, 1], got {self.explore_epsilon}")
        self._explore_random = 0
        self._explore_total = 0
        print(f"[EXPLORE] epsilon={self.explore_epsilon:.3f} (training data only)")
'''

anchor_act = '            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)\n'
new_act = '''            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)
            # EPSILON_EXPLORATION_PATCH_V1: alter the action that WILL be executed
            # on the next env step. Keep RSSM prev_action identical to that action.
            # This preserves correct observation/action alignment in replay.
            if self.explore_epsilon > 0.0:
                import torch.nn.functional as F
                mask = (torch.rand(act.shape[0], device=act.device) < self.explore_epsilon) & (~done)
                random_idx = torch.randint(act.shape[-1], (act.shape[0],), device=act.device)
                random_action = F.one_hot(random_idx, num_classes=act.shape[-1]).to(act.dtype)
                act = torch.where(mask[:, None], random_action, act)
                agent_state["prev_action"] = act.clone()
                self._explore_random += int(mask.sum().item())
                self._explore_total += int((~done).sum().item())
'''

anchor_log = '                    self.logger.scalar("train/opt/updates", update_count)\n'
new_log = '''                    self.logger.scalar("train/opt/updates", update_count)
                    if self._explore_total:
                        self.logger.scalar("train/explore_random_frac", self._explore_random / self._explore_total)
'''

for label, anchor in [('__init__',anchor_init), ('train act', anchor_act), ('logging', anchor_log)]:
    if s.count(anchor) != 1:
        raise SystemExit(f'[ERROR] Expected exactly one {label} anchor; found {s.count(anchor)}. Trainer unchanged.')

s = s.replace(anchor_init, new_init, 1)
s = s.replace(anchor_act, new_act, 1)
s = s.replace(anchor_log, new_log, 1)
compile(s, str(p), 'exec')
backup = p.with_name('trainer.py.before_epsilon_exploration')
if backup.exists():
    raise SystemExit(f'[ERROR] Backup already exists: {backup}; no files modified.')
shutil.copy2(p, backup)
p.write_text(s, encoding='utf-8')
print('[OK] trainer.py patched, backup:', backup)
print('[OK] Pass +trainer.explore_epsilon=0.05 in Hydra CLI.')
