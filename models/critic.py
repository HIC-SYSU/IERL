import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ContinuousCritic


class DroQNetwork(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout_rate: float = 0.01,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(last_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            ]
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DroQCritic(ContinuousCritic):

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch: list = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        features_dim: Optional[int] = None,
        activation_fn=nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 5,
        share_features_extractor: bool = False,
        dropout_rate: float = 0.05,
        **kwargs,
    ):
        if features_extractor is None:
            extract_class = kwargs.pop("features_extractor_class", None)
            extract_kwargs = kwargs.pop("features_extractor_kwargs", {})
            if extract_class is not None:
                features_extractor = extract_class(observation_space, **extract_kwargs)
                features_dim = features_extractor.features_dim

        super().__init__(
            observation_space,
            action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
            **kwargs,
        )

        self.features_dim = features_dim
        action_dim = int(np.prod(action_space.shape))
        input_dim = self.features_dim + action_dim

        self.q_networks = nn.ModuleList([
            DroQNetwork(input_dim, hidden_dims=[256, 256], dropout_rate=dropout_rate)
            for _ in range(n_critics)
        ])

        self.register_buffer(
            "ensemble_weights",
            torch.ones(n_critics) / n_critics,
        )

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        with torch.set_grad_enabled(True):
            features = self.extract_features(obs, self.features_extractor)
            q_input = torch.cat([features, actions], dim=1)
            return tuple(q_net(q_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.extract_features(obs, self.features_extractor)
            return self.q_networks[0](torch.cat([features, actions], dim=1))

    def update_ensemble_weights(
        self, obs: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor
    ) -> None:
        with torch.no_grad():
            features = self.extract_features(obs, self.features_extractor)
            q_input  = torch.cat([features, actions], dim=1)
            r        = returns.view(-1, 1)
            ss_tot   = ((r - r.mean()) ** 2).sum().clamp(min=1e-8)

            r2_scores = []
            for q_net in self.q_networks:
                preds  = q_net(q_input)
                ss_res = ((preds - r) ** 2).sum()
                r2_scores.append((1.0 - ss_res / ss_tot).item())

            r2_tensor = torch.tensor(r2_scores, device=self.ensemble_weights.device)
            self.ensemble_weights = torch.softmax(r2_tensor, dim=0)

    def weighted_q(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        features  = self.extract_features(obs, self.features_extractor)
        q_input   = torch.cat([features, actions], dim=1)
        q_values  = torch.stack(
            [q_net(q_input) for q_net in self.q_networks], dim=0
        )
        weights = self.ensemble_weights.view(-1, 1, 1)
        return (weights * q_values).sum(dim=0)
