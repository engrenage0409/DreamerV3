"""Evaluate an existing checkpoint and save HD Crafter MP4, no training."""
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer
from tools import Logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='logdir/crafter_5k_test')
    parser.add_argument('--episodes', type=int, default=1)
    args = parser.parse_args()
    run = Path(args.run)
    cfgpath = run / '.hydra' / 'config.yaml'
    ckptpath = run / 'latest.pt'
    if not cfgpath.exists():
        raise FileNotFoundError(f'Missing Hydra config: {cfgpath}; look for your original .hydra/config.yaml')
    if not ckptpath.exists():
        raise FileNotFoundError(f'Missing checkpoint: {ckptpath}')
    cfg = OmegaConf.load(cfgpath)
    cfg.env.eval_episode_num = args.episodes
    cfg.model.compile = False
    cfg.trainer.video_pred_log = False
    cfg.trainer.eval_episode_num = args.episodes
    cfg.trainer.eval_every = 1
    cfg.trainer.steps = 1
    cfg.trainer.update_log_every = 1
    out = run / 'hud_replay'
    out.mkdir(parents=True, exist_ok=True)
    print('Preparing evaluation environment...')
    train_envs, eval_envs, obs_space, act_space = make_envs(cfg.env)
    agent = Dreamer(cfg.model, obs_space, act_space).to(cfg.device)
    checkpoint = torch.load(ckptpath, map_location=cfg.device, weights_only=False)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    print(f'Loaded: {ckptpath}')
    logger = Logger(out)
    trainer = OnlineTrainer(cfg.trainer, None, logger, out, train_envs, eval_envs)
    try:
        with torch.no_grad():
            trainer.eval(agent, 5000)
    finally:
        for group in (eval_envs, train_envs):
            if group is not None:
                for worker in group.envs:
                    worker.close()
    print(f'Finished. MP4: {out / "hud_videos"}')


if __name__ == '__main__':
    main()
