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
    # Phase 5.3b: the ensemble is frozen and excluded from Dreamer checkpoints.
    # Attach AFTER resume so loading old checkpoint with strict=True still works.
    intrinsic_beta = float(config.get("intrinsic_beta", 0.0))
    if intrinsic_beta > 0.0:
        from disagreement_ensemble import DisagreementEnsemble
        ensemble_path = pathlib.Path(str(config.get("intrinsic_ensemble_path", ""))).expanduser()
        if not ensemble_path.is_file():
            raise FileNotFoundError(f"Intrinsic ensemble checkpoint missing: {ensemble_path}")
        # Only load a trusted .pt file created by this project.
        ens_state = torch.load(ensemble_path, map_location="cpu", weights_only=True)
        ensemble = DisagreementEnsemble(
            feature_dim=int(ens_state["feature_dim"]),
            action_dim=int(ens_state["action_dim"]),
            target_dim=int(ens_state["target_dim"]),
            hidden_dim=int(ens_state["hidden_dim"]),
            num_heads=int(ens_state["num_heads"]),
            seed=int(ens_state["seed"]),
        )
        if ensemble.feature_dim != agent.rssm.feat_size or ensemble.action_dim != agent.act_dim:
            raise ValueError("Ensemble dimensions differ from active Dreamer model")
        ensemble.load_state_dict(ens_state["ensemble_state_dict"], strict=True)
        agent.configure_intrinsic(
            ensemble, beta=intrinsic_beta,
            scale=float(config.get("intrinsic_scale", 0.01035)),
            clip=float(config.get("intrinsic_clip", 3.0)),
        )
        print(f"[INTRINSIC] beta={intrinsic_beta:g}, scale={agent.intrinsic_scale:g}, "
              f"clip={agent.intrinsic_clip:g}, checkpoint={ensemble_path}", flush=True)
    else:
        print("[INTRINSIC] disabled (control)", flush=True)

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
