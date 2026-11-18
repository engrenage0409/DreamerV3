import atexit
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from checkpointing import save_checkpoint, load_checkpoint  # CHECKPOINT_PATCH_V1
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    checkpoint_every = int(config.get("checkpoint_every", 5000))
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


if __name__ == "__main__":
    main()
