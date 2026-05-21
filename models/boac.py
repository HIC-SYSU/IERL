import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from stable_baselines3 import SAC

from .critic import DroQCritic


class BOAC(SAC):

    def __init__(
        self,
        policy,
        env,
        gp_buffer_size: int   = 100,
        gp_fit_interval: int  = 200,
        gp_start_timesteps: int = 1000,
        bayes_lr: float       = 0.05,
        gp_k1_constant_value: float = 1.0,
        gp_k1_constant_bounds = (1e-5, 1e5),
        gp_length_scale: float = 1.0,
        gp_length_scale_bounds = (1e-2, 1e2),
        gp_k2_constant_value: float = 1e-5,
        gp_k2_constant_bounds = (1e-6, 1e6),
        gp_alpha: float = 1e-2,
        **kwargs,
    ):
        if "policy_kwargs" not in kwargs:
            kwargs["policy_kwargs"] = {}
        pk = kwargs["policy_kwargs"]
        pk.setdefault("share_features_extractor", False)
        ck = pk.setdefault("critic_kwargs", {})
        ck.setdefault("n_critics", 5)
        ck.setdefault("dropout_rate", 0.02)
        ck.setdefault("net_arch", [256, 256])
        super().__init__(policy, env, **kwargs)

        self.gp_buffer_size     = gp_buffer_size
        self.gp_fit_interval    = gp_fit_interval
        self.gp_start_timesteps = gp_start_timesteps
        self.bayes_lr           = bayes_lr

        self.param_history = []
        self.return_history = []

        self.kernel = (
            C(gp_k1_constant_value, gp_k1_constant_bounds) * RBF(length_scale=gp_length_scale, length_scale_bounds=gp_length_scale_bounds)
            + C(gp_k2_constant_value, gp_k2_constant_bounds)
        )
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=2,
            alpha=gp_alpha,
            normalize_y=True,
        )
        self.target_layer_name = "mu"

        self._bayes_optimizer = None

        self._current_rewards: np.ndarray | None = None

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps, batch_size)
        if self.replay_buffer.size() >= batch_size:
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            self.critic.update_ensemble_weights(
                data.observations, data.actions, data.rewards
            )
        if (
            self.num_timesteps > self.gp_start_timesteps
            and (self.num_timesteps // self.gp_fit_interval) != ((self.num_timesteps - self.n_envs) // self.gp_fit_interval)
        ):
            self._apply_bayesian_correction()

    def _store_transition(self, replay_buffer, buffer_actions, new_obs, rewards, dones, infos):
        self._current_rewards = np.asarray(rewards, dtype=np.float32)
        super()._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

    def _on_step(self) -> None:
        super()._on_step()
        if self._current_rewards is None:
            return

        theta = self._get_target_params_numpy()
        current_return = float(np.mean(self._current_rewards))

        self.param_history.append(theta)
        self.return_history.append(current_return)
        if len(self.param_history) > self.gp_buffer_size:
            self.param_history.pop(0)
            self.return_history.pop(0)

    def _get_target_params_numpy(self) -> np.ndarray:
        return self.actor.mu.weight.data.cpu().numpy().flatten()

    def _apply_bayesian_correction(self) -> None:
        if len(self.param_history) < 20:
            return

        X = np.array(self.param_history)
        y = np.array(self.return_history)

        try:
            self.gp.fit(X, y)
        except Exception:
            return

        rbf_kernel = self._find_rbf_kernel(self.gp.kernel_)
        if rbf_kernel is None:
            return

        length_scale = rbf_kernel.length_scale
        alpha = self.gp.alpha_
        X_train = self.gp.X_train_

        theta_curr = self._get_target_params_numpy().reshape(1, -1)
        dists = np.sum((theta_curr - X_train) ** 2, axis=1)
        k_values = np.exp(-0.5 * dists / (length_scale ** 2))

        diffs = X_train - theta_curr
        weights = (alpha * k_values).reshape(-1, 1)
        gp_grad = np.sum(weights * diffs, axis=0) / (length_scale ** 2)

        grad_norm = np.linalg.norm(gp_grad)
        if grad_norm > 1.0:
            gp_grad /= grad_norm

        if self._bayes_optimizer is None:
            self._bayes_optimizer = torch.optim.Adam(
                [self.actor.mu.weight], lr=self.bayes_lr
            )

        gp_grad_tensor = (
            torch.from_numpy(gp_grad)
            .float()
            .to(self.actor.device)
            .view(self.actor.mu.weight.shape)
        )
        self.actor.mu.weight.grad = -gp_grad_tensor
        self._bayes_optimizer.step()
        self._bayes_optimizer.zero_grad()

    @staticmethod
    def _find_rbf_kernel(kernel):
        stack = [kernel]
        while stack:
            k = stack.pop()
            if isinstance(k, RBF):
                return k
            if hasattr(k, "k1"):
                stack.append(k.k1)
            if hasattr(k, "k2"):
                stack.append(k.k2)
        return None
