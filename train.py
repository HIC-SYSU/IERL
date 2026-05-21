import os
import sys
import yaml
import random
from functools import partial
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from environment import NonDaemonSubprocVecEnv, StentPlacementEnv
from models import GPASE, BOAC, DroQCritic, DroQPolicy, PersistedReplayBuffer
from simulation.interface import SimulationInterface
from dataLoader import VascularDataset



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def match_percent(pred: np.ndarray, ref: np.ndarray, tol: float = 0.1) -> float:
    ratio = np.abs(pred - ref) / (np.abs(ref) + 1e-8)
    return float(np.mean(ratio < tol))


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)




class MatchStatsCallback(BaseCallback):

    def __init__(self, match_threshold: float = 0.85, tol: float = 0.1, verbose: int = 0):
        super().__init__(verbose)
        self.match_threshold = match_threshold
        self.tol = tol
        self._reached = False

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "pred_mm" in info and "ref_mm" in info:
                rate = match_percent(info["pred_mm"], info["ref_mm"], self.tol)
                if not self._reached and rate >= self.match_threshold:
                    print(f"[Callback] {self.match_threshold*100:.0f}% match reached at step {self.num_timesteps}")
                    self._reached = True
        return True


class LossPlotCallback(BaseCallback):

    def __init__(self, save_path: str = "loss_curve.png", verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []

    def _on_step(self) -> bool:
        log = self.model.logger.name_to_value
        if (al := log.get("train/actor_loss")) is not None:
            self.actor_losses.append(al)
        if (cl := log.get("train/critic_loss")) is not None:
            self.critic_losses.append(cl)
        return True

    def _on_training_end(self) -> None:
        plt.figure(figsize=(8, 5))
        plt.plot(self.actor_losses, label="Actor Loss")
        plt.plot(self.critic_losses, label="Critic Loss")
        plt.title("IERL Training Loss")
        plt.xlabel("Steps (approx)")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_path)
        if self.verbose:
            print(f"[Callback] Loss plot saved → {self.save_path}")


def train(config: Dict[str, Any]) -> None:
    set_seed(config.get("seed", 42))

    run_dir = config.get("run_dir", "runs/default")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    print("Loading vessel data...")

    data_dir = config["data_dir"]
    val_ratio = config.get("val_ratio", 0.2)
    seq_len      = config.get("seq_len", 100)
    features_dim = config.get("features_dim", 128)
    num_heads    = config.get("num_heads", 4)

    train_dataset = VascularDataset(data_dir=data_dir, split="train",
                                    val_ratio=val_ratio, seed=config["seed"], seq_len=seq_len)
    val_dataset   = VascularDataset(data_dir=data_dir, split="val",
                                    val_ratio=val_ratio, seed=config["seed"], seq_len=seq_len)

    num_envs = config.get("num_envs", 16)
    total_timesteps = config["num_epochs"] * len(train_dataset) * num_envs
    print(f"Dataset: {len(train_dataset)} train / {len(val_dataset)} val samples")

    vessels = train_dataset

    env_cfg = config.get("env", {})
    sim_start_method = config.get("sim_start_method", None)
    vec_env_start_method = config.get("vec_env_start_method", None)
    sim_interface_factory = (
        partial(SimulationInterface, start_method=sim_start_method)
        if sim_start_method is not None
        else SimulationInterface
    )

    def make_env():
        def _init():
            env = StentPlacementEnv(
                vessels,
                L=seq_len,
                alpha1=env_cfg.get("alpha1", 1.0),
                alpha2=env_cfg.get("alpha2", 0.5),
                lam=env_cfg.get("lam", 1.0),
                omega=env_cfg.get("omega", 0.1),
                lam1=env_cfg.get("lam1", 0.5),
                lam2=env_cfg.get("lam2", 0.5),
                q_hyperemia_ml_s=env_cfg.get("q_hyperemia_ml_s", 1.0),
                p_inlet_mmhg=env_cfg.get("p_inlet_mmhg", 100.0),
                stent_diameters_mm=env_cfg.get("stent_diameters_mm", None),
                stent_lengths_mm=env_cfg.get("stent_lengths_mm", None),
                meta_cache_size=env_cfg.get("meta_cache_size", 64),
                sim_interface=sim_interface_factory,
            )
            return Monitor(env)
        return _init

    if num_envs > 1:
        venv = NonDaemonSubprocVecEnv(
            [make_env() for _ in range(num_envs)],
            start_method=vec_env_start_method,
            daemon=False,
        )
    else:
        venv = DummyVecEnv([make_env()])

    def make_val_env():
        def _init():
            env = StentPlacementEnv(
                val_dataset,
                L=seq_len,
                alpha1=env_cfg.get("alpha1", 1.0),
                alpha2=env_cfg.get("alpha2", 0.5),
                lam=env_cfg.get("lam", 1.0),
                omega=env_cfg.get("omega", 0.1),
                lam1=env_cfg.get("lam1", 0.5),
                lam2=env_cfg.get("lam2", 0.5),
                q_hyperemia_ml_s=env_cfg.get("q_hyperemia_ml_s", 1.0),
                p_inlet_mmhg=env_cfg.get("p_inlet_mmhg", 100.0),
                stent_diameters_mm=env_cfg.get("stent_diameters_mm", None),
                stent_lengths_mm=env_cfg.get("stent_lengths_mm", None),
                meta_cache_size=env_cfg.get("meta_cache_size", 64),
                sim_interface=sim_interface_factory,
            )
            return Monitor(env)
        return _init

    if num_envs > 1:
        val_venv = NonDaemonSubprocVecEnv(
            [make_val_env()],
            start_method=vec_env_start_method,
            daemon=False,
        )
    else:
        val_venv = DummyVecEnv([make_val_env()])

    policy_kwargs = dict(
        features_extractor_class=GPASE,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
            num_heads=num_heads,
            seq_len=seq_len,
        ),
        net_arch=config["net_arch"],
        critic_kwargs=dict(
            n_critics=config["n_critics"],
            dropout_rate=config["dropout_rate"],
            net_arch=config["net_arch"],
        ),
        share_features_extractor=False,
    )



    tfreq = config["train_freq"]
    if isinstance(tfreq, list):
        tfreq = tuple(tfreq)

    try:
        model = BOAC(
            policy=DroQPolicy,
            env=venv,
            policy_kwargs=policy_kwargs,
            learning_rate=config["learning_rate"],

            replay_buffer_class=PersistedReplayBuffer,
            replay_buffer_kwargs=dict(persist_path=os.path.join(run_dir, "buffer")),

            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            train_freq=tfreq,
            gradient_steps=config["gradient_steps"],

            seed=config["seed"],
            gamma=config.get("gamma", 0.99),
            verbose=1,

            gp_buffer_size=config.get("gp_buffer_size", 100),
            gp_fit_interval=config.get("gp_fit_interval", 200),
            gp_start_timesteps=config.get("gp_start_timesteps", 1000),
            bayes_lr=config.get("bayes_lr", 0.05),

            gp_k1_constant_value=config.get("gp_k1_constant_value", 1.0),
            gp_k1_constant_bounds=config.get("gp_k1_constant_bounds", (1e-5, 1e5)),
            gp_length_scale=config.get("gp_length_scale", 1.0),
            gp_length_scale_bounds=config.get("gp_length_scale_bounds", (1e-2, 1e2)),
            gp_k2_constant_value=config.get("gp_k2_constant_value", 1e-5),
            gp_k2_constant_bounds=config.get("gp_k2_constant_bounds", (1e-6, 1e6)),
            gp_alpha=config.get("gp_alpha", 1e-2),
        )

        if isinstance(model.policy.critic, DroQCritic):
            print(f"DroQCritic integrated — {len(model.policy.critic.q_networks)} ensemble critics.")
        else:
            print("Warning: DroQCritic not detected in policy.")

        best_model_dir  = os.path.join(run_dir, "best_model")
        checkpoint_dir  = os.path.join(run_dir, "checkpoints")
        eval_callback = EvalCallback(
            val_venv,
            eval_freq=config.get("eval_freq", 5000),
            n_eval_episodes=config.get("n_eval_episodes", 20),
            deterministic=True,
            best_model_save_path=best_model_dir,
            log_path=best_model_dir,
            verbose=1,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=config.get("checkpoint_freq", 10000),
            save_path=checkpoint_dir,
            name_prefix="boac",
            verbose=1,
        )
        callbacks = [
            eval_callback,
            checkpoint_callback,
            MatchStatsCallback(match_threshold=config.get("match_threshold", 0.85), verbose=1),
            LossPlotCallback(save_path=os.path.join(run_dir, "loss_curve.png"), verbose=1),
        ]

        print(f"Training for {config['num_epochs']} epochs → {total_timesteps} total timesteps")
        model.learn(total_timesteps=total_timesteps, callback=callbacks)

        save_path = os.path.join(run_dir, "model_final")
        model.save(save_path)
        print(f"Training complete. Model saved → {save_path}")
    finally:
        venv.close()
        val_venv.close()


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "train_config.yaml"
    print(f"Loading config: {cfg_file}")
    train(load_config(cfg_file))
