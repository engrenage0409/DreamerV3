"""Patch r2dreamer train.py/trainer.py to save and resume training.

Run in repo root: python install_checkpoint.py
Preserves existing ETA and HUD patches. Re-running is a no-op.
"""
from pathlib import Path
import shutil
import py_compile

root = Path(__file__).resolve().parent
train_path = root / 'train.py'
trainer_path = root / 'trainer.py'
if not (train_path.exists() and trainer_path.exists() and (root / 'checkpointing.py').exists()):
    raise SystemExit('Copy install_checkpoint.py and checkpointing.py to the r2dreamer root first.')

train = train_path.read_text(encoding='utf-8')
trainer = trainer_path.read_text(encoding='utf-8')
if 'CHECKPOINT_PATCH_V1' in train or 'CHECKPOINT_PATCH_V1' in trainer:
    if 'CHECKPOINT_PATCH_V1' in train and 'CHECKPOINT_PATCH_V1' in trainer:
        print('Checkpoint patch already installed.')
        raise SystemExit(0)
    raise SystemExit('Partial checkpoint patch detected; restore .checkpoint_backup files and retry.')

# Safely patch the original train entry point.
assert train.count('import tools\n') == 1, 'train.py imports changed'
train = train.replace('import tools\n', 'import tools\nfrom checkpointing import save_checkpoint, load_checkpoint  # CHECKPOINT_PATCH_V1\n', 1)
original = '''    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")
'''
replacement = '''    checkpoint_every = int(config.get("checkpoint_every", 5000))
    policy_trainer = OnlineTrainer(
        config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs,
        checkpoint_every=checkpoint_every,
        checkpoint_callback=lambda agent, trainer, step, updates: save_checkpoint(
            logdir, agent, replay_buffer, trainer, step, updates, numbered=True
        ),
    )
    resume_path = str(config.get("resume", "")).strip()
    if resume_path:
        resume_state = load_checkpoint(resume_path, agent, replay_buffer)
        policy_trainer.restore_training_state(resume_state)
    final_step, final_updates = policy_trainer.begin(agent)
    save_checkpoint(logdir, agent, replay_buffer, policy_trainer,
                    final_step, final_updates, numbered=False)
    # One final evaluation video at the terminal step. Latest checkpoint is
    # already safe on disk even if video rendering fails.
    if int(config.trainer.eval_episode_num) > 0 and eval_envs is not None:
        try:
            policy_trainer.eval(agent, final_step)
        except Exception as err:
            print(f"[WARNING] Final evaluation failed, checkpoint is saved: {err}")
'''
assert train.count(original) == 1, 'train.py model saving block changed'
train = train.replace(original, replacement, 1)

old_sig = '    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs):\n'
new_sig = '    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs, checkpoint_every=0, checkpoint_callback=None):\n'
assert trainer.count(old_sig) == 1, 'OnlineTrainer signature changed'
trainer = trainer.replace(old_sig, new_sig, 1)

old_assign = '        self.replay_buffer = replay_buffer\n'
new_assign = '''        # CHECKPOINT_PATCH_V1: supports full replay/optimizer restart.
        self.checkpoint_every = int(checkpoint_every)
        self._checkpoint_callback = checkpoint_callback
        self._resume_state = None
        self.replay_buffer = replay_buffer
'''
assert trainer.count(old_assign) == 1
trainer = trainer.replace(old_assign, new_assign, 1)

needle = '    def eval(self, agent, train_step):\n'
assert trainer.count(needle) == 1
trainer = trainer.replace(needle, '''    def restore_training_state(self, state):
        self._resume_state = dict(state)
        self._updates_needed._last = state["updates_last"]
        self._should_eval._last = state["eval_last"]
        self._should_log._last = state["log_last"]
        self._should_pretrain._once = state["pretrain_once"]

''' + needle, 1)

old_step = '''        step = self.replay_buffer.count() * self._action_repeat
        update_count = 0
'''
new_step = '''        step = (int(self._resume_state["step"]) if self._resume_state
                else self.replay_buffer.count() * self._action_repeat)
        update_count = (int(self._resume_state["update_count"]) if self._resume_state else 0)
        initial_step = step
        next_checkpoint = ((step // self.checkpoint_every) + 1) * self.checkpoint_every if self.checkpoint_every else None
'''
assert trainer.count(old_step) == 1
trainer = trainer.replace(old_step, new_step, 1)

# Maintain correct ETA if user previously installed the time progress display.
trainer = trainer.replace('speed = step / max(elapsed, 1e-6)', 'speed = (step - initial_step) / max(elapsed, 1e-6)')

# The final statement in original begin() is the metric log. HUD/ETA patches
# may append extra statements afterward, so insert checkpoint at the end of begin.
# This version of trainer.py has begin() as its final method, verified in the
# upstream NM512/r2dreamer source.
assert '    def begin(self, agent):\n' in trainer
assert not any(line.startswith('    def ') for line in trainer.split('    def begin(self, agent):\n', 1)[1].splitlines()), 'Unexpected method after begin()'

trainer = trainer.rstrip() + '''
            if (self._checkpoint_callback is not None and next_checkpoint is not None
                    and step >= next_checkpoint):
                self._checkpoint_callback(agent, self, step, update_count)
                next_checkpoint = ((step // self.checkpoint_every) + 1) * self.checkpoint_every

        return step, update_count
'''

# Check syntax BEFORE changing the user's files.
compile(train, str(train_path), 'exec')
compile(trainer, str(trainer_path), 'exec')
shutil.copy2(train_path, train_path.with_name('train.py.checkpoint_backup'))
shutil.copy2(trainer_path, trainer_path.with_name('trainer.py.checkpoint_backup'))
train_path.write_text(train, encoding='utf-8')
trainer_path.write_text(trainer, encoding='utf-8')
py_compile.compile(str(train_path), doraise=True)
py_compile.compile(str(trainer_path), doraise=True)
print('Installed checkpoint + full replay resume patch.')
print('Backups: train.py.checkpoint_backup and trainer.py.checkpoint_backup')
