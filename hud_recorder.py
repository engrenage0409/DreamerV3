"""HD Crafter MP4 HUD, independent of Dreamer training tensors."""
import atexit
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class HudRecorder:
    def __init__(self, path, fps=12, width=576, panel=320):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.panel = panel
        self.height = width
        self.count = 0
        self.total_reward = 0.0
        self._proc = None
        self.font = ImageFont.load_default()
        self._closed = False
        atexit.register(self.close)

    def _start(self):
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{self.width+self.panel}x{self.height}',
               '-r', str(self.fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
               '-crf', '21', '-pix_fmt', 'yuv420p', str(self.path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def add(self, image, inventory, achievements, action='START', reward=0.0, step=None):
        if self._closed:
            return
        if self._proc is None:
            self._start()
        self.total_reward += float(reward)
        self.count += 1
        frame = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert('RGB')
        if frame.size != (self.width, self.height):
            frame = frame.resize((self.width, self.height), Image.Resampling.NEAREST)
        canvas = Image.new('RGB', (self.width+self.panel, self.height), '#111923')
        canvas.paste(frame, (0, 0))
        d = ImageDraw.Draw(canvas)
        x = self.width + 16
        def line(y, text, fill='#e4e9f2'):
            d.text((x, y), str(text), font=self.font, fill=fill)
        line(14, f'CRAFTER / EPISODE REPLAY', '#82d5ff')
        line(42, f'Step: {self.count - 1 if step is None else step}')
        line(64, f'Action: {action}', '#82d5ff')
        line(86, f'Reward: {float(reward):+.2f}')
        line(108, f'Total Reward: {self.total_reward:+.2f}', '#9ee6b2')
        d.line([(x, 139), (self.width + self.panel - 16, 139)], fill='#475465', width=1)
        line(153, 'SURVIVAL', '#f1c477')
        for k, y, col in [('health',181,'#ec7272'), ('food',218,'#e4bb65'),
                          ('drink',255,'#66b9ed'), ('energy',292,'#83d8a0')]:
            value = inventory.get(k)
            line(y, f'{k.upper():8s}  {value if value is not None else "N/A"}')
            if isinstance(value, (int,float)):
                # Default Crafter caps at 9 for these attributes.
                d.rectangle((x, y+17, x+240, y+24), fill='#354150')
                d.rectangle((x, y+17, x+int(240*min(1,max(0,float(value)/9))), y+24), fill=col)
        d.line([(x, 330), (self.width+self.panel-16, 330)], fill='#475465')
        line(344, 'INVENTORY', '#f1c477')
        items = [(k,v) for k,v in inventory.items() if k not in {'health','food','drink','energy'} and v]
        for j, (k,v) in enumerate(items[:7]):
            line(369+18*j, f'{k}: {v}')
        d.line([(x, 503), (self.width+self.panel-16, 503)], fill='#475465')
        unlocked = [k for k,v in achievements.items() if v]
        line(513, 'ACHIEVEMENTS', '#f1c477')
        line(535, (', '.join(unlocked[:2]) or 'None yet')[:43])
        self._proc.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._proc is not None:
            self._proc.stdin.close()
            rc = self._proc.wait()
            if rc:
                raise RuntimeError(f'ffmpeg exited {rc}: {self.path}')
            print(f'[HUD VIDEO] Saved {self.path} ({self.count} frames)', flush=True)
