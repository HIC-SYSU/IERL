import torch
from typing import Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.sac.policies import SACPolicy

from .encoder import GPASE
from .critic import DroQCritic


class DroQPolicy(SACPolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=torch.nn.ReLU,
        use_sde=False,
        log_std_init=-3,
        use_expln=False,
        clip_mean=2.0,
        features_extractor_class=GPASE,
        features_extractor_kwargs=None,
        normalize_images=True,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
        n_critics=5,
        share_features_extractor=False,
        **kwargs,
    ):
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.actor_kwargs = {
            **self.net_args,
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }

        extra_critic_kwargs = kwargs.get("critic_kwargs", {})
        self.critic_kwargs = {
            **self.net_args,
            "n_critics": n_critics,
            **extra_critic_kwargs,
        }

        self.critic_class = DroQCritic
        self.share_features_extractor = share_features_extractor
        self._build(lr_schedule)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ):
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return self.critic_class(**critic_kwargs).to(self.device)
