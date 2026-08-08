"""DQN Q-network: vanilla, Double, and Dueling variants.

Controlled by flags in HyperParams config — same weights, different forward pass.
Input: feature vector of size 4 (or 6 with lookahead).
Output: Q-values for 2 actions (no_op=0, jump=1).
"""

import torch
import torch.nn as nn
from torch import Tensor


class QNetwork(nn.Module):
    """MLP Q-network supporting vanilla / Dueling architectures.

    Args:
        input_dim: Size of the feature vector (4 or 6).
        hidden_dim: Width of hidden layers.
        n_actions: Number of discrete actions (2 for GD cube).
        dueling: If True, use Dueling DQN head (V + A streams).
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        n_actions: int = 2,
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self._dueling = dueling

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if dueling:
            self.value_head = nn.Linear(hidden_dim, 1)
            self.adv_head = nn.Linear(hidden_dim, n_actions)
        else:
            self.q_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        """Return Q-values for each action.

        Args:
            x: Float tensor of shape (batch, input_dim).

        Returns:
            Q-values of shape (batch, n_actions).
        """
        h = self.backbone(x)
        if self._dueling:
            v = self.value_head(h)                    # (batch, 1)
            a = self.adv_head(h)                       # (batch, n_actions)
            return v + a - a.mean(dim=-1, keepdim=True)
        return self.q_head(h)
