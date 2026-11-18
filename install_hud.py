"""Run once from r2dreamer root: python install_hud.py"""
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parent
trainer_file = root / 'trainer.py'
wrapper_file = root / 'envs' / 'wrappers.py'
if not trainer_file.exists() or not wrapper_file.exists():
    raise SystemExit('Put install_hud.py and hud_recorder.py at D:\\GH\\r2dreamer first.')
if not (root / 'hud_recorder.py').exists():
    raise SystemExit('hud_recorder.py missing.')

w = wrapper_file.read_text(encoding='utf-8')
if 'def video_snapshot(self' not in w:
    target = 'class Dtype(gym.Wrapper):\n'
    assert w.count(target) == 1, 'Dtype class not found'
    new = '''class Dtype(gym.Wrapper):
    def video_snapshot(self, size=(576, 576)):
        """Retrieve HD Crafter frame and real state from the environment worker.

        Called only for evaluation recording; never changes agent observations.
        """
        inner = self.env
        while hasattr(inner, 'env'):
            inner = inner.env
        native = getattr(inner, '_env', None)
        if native is None or not hasattr(native, '_player'):
            raise RuntimeError('video_snapshot only supports Crafter.')
        player = native._player
        return (native.render(size=size), dict(player.inventory),
                dict(player.achievements), list(native.action_names))

'''
    w = w.replace(target, new, 1)
    shutil.copy2(wrapper_file, wrapper_file.with_suffix('.py.hud_backup'))
    wrapper_file.write_text(w, encoding='utf-8')
    print('Patched envs/wrappers.py')
else:
    print('envs/wrappers.py already patched')

t = trainer_file.read_text(encoding='utf-8')
if 'HUD_VIDEO_PATCH_V1' not in t:
    if 'from pathlib import Path\n' not in t:
        t = 'from pathlib import Path\n' + t
    t = t.replace('import tools\n', 'import tools\nfrom hud_recorder import HudRecorder\n', 1)
    signature = '        self.replay_buffer = replay_buffer\n'
    assert t.count(signature) == 1
    t = t.replace(signature, '        self.logdir = Path(logdir)\n' + signature, 1)
    setup = '        agent_state = agent.get_initial_state(envs.env_num)\n'
    assert t.count(setup) == 2
    # The first occurrence is inside eval(), second inside begin().
    t = t.replace(setup,
        '''        # HUD_VIDEO_PATCH_V1: streamed MP4 from the native environment renderer.
        hud = HudRecorder(self.logdir / 'hud_videos' / f'eval_step_{train_step}.mp4')
''' + setup, 1)
    anchor = '''            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
'''
    assert t.count(anchor) >= 1, 'eval env step block changed: patch needs review'
    t = t.replace(anchor,
        '''            was_reset = bool(done[0].item())
            action_index = int(act[0].argmax().item())
            trans, step_done = envs.step(act.detach(), done)
            if hud is not None and not bool(once_done[0].item()):
                snapshot, inventory, achievements, action_names = envs.envs[0].video_snapshot()
                name = 'START' if was_reset else action_names[action_index]
                hud.add(snapshot, inventory, achievements, action=name,
                        reward=float(trans['reward'][0, 0].item()))
                if bool(step_done[0].item()):
                    hud.close()
                    hud = None
            # dict of (B, 1, *)
''', 1)
    # Close if loop was interrupted by a time limit or nonstandard env.
    anchor2 = '        # dict of (B, T, *)\n        cache = torch.stack(cache, dim=1) if len(cache) else None\n'
    assert t.count(anchor2) == 1
    t = t.replace(anchor2,
        '''        if hud is not None:
            hud.close()
''' + anchor2, 1)
    shutil.copy2(trainer_file, trainer_file.with_suffix('.py.hud_backup'))
    trainer_file.write_text(t, encoding='utf-8')
    print('Patched trainer.py')
else:
    print('trainer.py already patched')

import py_compile
py_compile.compile(str(trainer_file), doraise=True)
py_compile.compile(str(wrapper_file), doraise=True)
print('HUD patch installed, syntax validated.')
