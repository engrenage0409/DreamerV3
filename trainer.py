from pathlib import Path
import time
from datetime import timedelta

import torch

import tools
from hud_recorder import HudRecorder


class OnlineTrainer:
    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs, checkpoint_every=0, checkpoint_callback=None):
        self.logdir = Path(logdir)
        # CHECKPOINT_PATCH_V1: supports full replay/optimizer restart.
        self.checkpoint_every = int(checkpoint_every)
        self._checkpoint_callback = checkpoint_callback
        self._resume_state = None
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat
        # EPSILON_EXPLORATION_PATCH_V1: epsilon-random TRAIN collection only.
        self.explore_epsilon = float(getattr(config, "explore_epsilon", 0.0))
        if not (0.0 <= self.explore_epsilon <= 1.0):
            raise ValueError(f"explore_epsilon must be in [0, 1], got {self.explore_epsilon}")
        self._explore_random = 0
        self._explore_total = 0
        print(f"[EXPLORE] epsilon={self.explore_epsilon:.3f} (training data only)")

    def restore_training_state(self, state):
        self._resume_state = dict(state)
        self._updates_needed._last = state["updates_last"]
        self._should_eval._last = state["eval_last"]
        self._should_log._last = state["log_last"]
        self._should_pretrain._once = state["pretrain_once"]

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        For CPU-based environments (``ParallelEnv``), stepping is executed on
        CPU and observations are moved to GPU asynchronously.  For GPU-resident
        environments (``IsaacLabVecEnv``), no device transfer is needed —
        ``.to()`` is a no-op when source and target devices match.
        """
        print("Evaluating the policy...")
        envs = self.eval_envs
        agent.eval()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        log_metrics = {}
        # cache is only used for video logging / open-loop prediction.
        cache = []
        # HUD_VIDEO_PATCH_V1: streamed MP4 from the native environment renderer.
        hud = HudRecorder(self.logdir / 'hud_videos' / f'eval_step_{train_step}.mp4')
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            was_reset = bool(done[0].item())
            action_index = int(act[0].argmax().item())
            trans, step_done = envs.step(act.detach(), done)
            if hud is not None and not bool(once_done[0].item()):
                snapshot, inventory, achievements, action_names = envs.envs[0].video_snapshot()()
                name = 'START' if was_reset else action_names[action_index]
                hud.add(snapshot, inventory, achievements, action=name,
                        reward=float(trans['reward'][0, 0].item()))
                if bool(step_done[0].item()):
                    hud.close()
                    hud = None
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # Store transition.
            # We keep the observation and the action that produced it together.
            trans["action"] = act
            if len(cache) < 1000:
                cache.append(trans.clone())
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                    log_metrics[key] += value[:, 0] * ~once_done
            once_done |= done
        if hud is not None:
            hud.close()
        # dict of (B, T, *)
        cache = torch.stack(cache, dim=1) if len(cache) else None
        self.logger.scalar("episode/eval_score", returns.mean())
        self.logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
        for key, value in log_metrics.items():
            if key == "log_success":
                value = torch.clip(value, max=1.0)  # make sure 1.0 for success episode
            self.logger.scalar(f"episode/eval_{key[4:]}", value.mean())
        if cache is not None and "image" in cache:
            self.logger.video("eval_video", tools.to_np(cache["image"][:1]))
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            self.logger.video(
                "eval_open_loop",
                tools.to_np(
                    agent.video_pred(
                        cache[:1],  # give only first batch
                        (initial["stoch"], initial["deter"]),
                    )
                ),
            )
        self.logger.write(train_step)
        agent.train()

    def begin(self, agent):
        """Main online training loop.

        For CPU-based environments the loop overlaps CPU stepping and GPU
        model execution via pinned-memory async H2D transfers.  For
        GPU-resident environments (IsaacLab) no transfer is needed —
        ``.to()`` is a no-op when the data is already on the target device.
        """
        envs = self.train_envs
        video_cache = []
        step = (int(self._resume_state["step"]) if self._resume_state
                else self.replay_buffer.count() * self._action_repeat)
        update_count = (int(self._resume_state["update_count"]) if self._resume_state else 0)
        initial_step = step
        next_checkpoint = ((step // self.checkpoint_every) + 1) * self.checkpoint_every if self.checkpoint_every else None
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_ids = torch.arange(
            envs.env_num, dtype=torch.int32, device=agent.device
        )  # Kept constant so short episodes (< batch_length) remain sampable; RSSM resets via is_first.
        train_metrics = tools.MetricAccumulator()
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        start_time = time.monotonic()
        while step < self.steps:
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0 and self.eval_envs is not None:
                self.eval(agent, step)
            # Save metrics
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        if i == 0 and len(video_cache) > 0:
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video("train_video", tools.to_np(video[None]))
                            video_cache = []
                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        self.logger.write(step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
            step += int((~done).sum()) * self._action_repeat  # step is based on env side
            lengths += ~done

            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)
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

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids  # Don't lift dim
            if "image" in trans:
                video_cache.append(trans["image"][0])
            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]
            # Update models after enough data has accumulated
            if step // (envs.env_num * self._action_repeat) > self.batch_length + 1:
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)
                for _ in range(update_num):
                    train_metrics.add(agent.update(self.replay_buffer))
                update_count += update_num
                # Log training metrics
                if self._should_log(step):
                    for name, value in train_metrics.mean().items():
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", update_count)
                    if self._explore_total:
                        self.logger.scalar("train/explore_random_frac", self._explore_random / self._explore_total)
                    if self.video_pred_log:
                        data, _, initial = self.replay_buffer.sample(prefer_online=False)
                        self.logger.video("open_loop", tools.to_np(agent.video_pred(data, initial)))
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(name, tools.to_np(param))
                    self.logger.write(step, fps=True)
                    elapsed = time.monotonic() - start_time
                    speed = (step - initial_step) / max(elapsed, 1e-6)
                    remaining = max(0, self.steps - step)
                    eta = remaining / max(speed, 1e-6)
                    print(
                        f"[PROGRESS] {step}/{self.steps} "
                        f"({100 * step / self.steps:.1f}%) | "
                        f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                        f"ETA: {timedelta(seconds=int(eta))} | "
                        f"Avg FPS: {speed:.2f}",
                        flush=True,
                    )
            if (self._checkpoint_callback is not None and next_checkpoint is not None
                    and step >= next_checkpoint):
                self._checkpoint_callback(agent, self, step, update_count)
                next_checkpoint = ((step // self.checkpoint_every) + 1) * self.checkpoint_every

        return step, update_count
