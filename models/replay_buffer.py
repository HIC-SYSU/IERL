import os
import pickle
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

class PersistedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device="auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        persist_path: str = None,
        **kwargs
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.persist_path = persist_path

        if self.persist_path:
            os.makedirs(self.persist_path, exist_ok=True)

    def add(self, obs, next_obs, action, reward, done, infos):
        super().add(obs, next_obs, action, reward, done, infos)

        if self.persist_path:
            self._save_to_disk(obs, action, reward, next_obs, done)

    def _save_to_disk(self, obs, action, reward, next_obs, done):
        experience = (
            np.array(obs, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(done, dtype=np.bool_)
        )
        file_path = os.path.join(self.persist_path, 'replay.pkl')
        with open(file_path, 'ab') as f:
            pickle.dump(experience, f)

    def clear_all(self):
        self.reset()
