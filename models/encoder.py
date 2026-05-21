import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GPASE(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        num_heads: int = 4,
        seq_len: int = 100,
    ):
        super().__init__(observation_space, features_dim)

        self.seq_len = seq_len
        self.input_channels = 4

        self.embedding = nn.Linear(self.input_channels, features_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, features_dim))

        self.mha = nn.MultiheadAttention(
            embed_dim=features_dim, num_heads=num_heads, batch_first=True
        )

        self.layer_norm1 = nn.LayerNorm(features_dim)
        self.ffn = nn.Sequential(
            nn.Linear(features_dim, features_dim * 2),
            nn.ReLU(),
            nn.Linear(features_dim * 2, features_dim),
        )
        self.layer_norm2 = nn.LayerNorm(features_dim)

        self.final_projection = nn.Linear(features_dim * seq_len, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        x = observations.view(batch_size, self.seq_len, self.input_channels)

        x = self.embedding(x) + self.pos_embedding

        attn_output, _ = self.mha(x, x, x)
        x = self.layer_norm1(x + attn_output)

        x = self.layer_norm2(x + self.ffn(x))

        x = self.final_projection(x.flatten(start_dim=1))
        return x
