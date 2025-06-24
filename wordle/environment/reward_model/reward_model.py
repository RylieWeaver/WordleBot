# General
from math import sqrt

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F


############################################
# NETWORK
############################################
class RewardNet(nn.Module):
    def __init__(self, state_size, hidden_dim=128, layers=3, dropout=0.1):
        super().__init__()
        state_size = 26*11
        self.network = nn.ModuleList()
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.vocab_entropy = torch.log2(torch.tensor(2315.0))

        # Embedding layers
        self.embed = nn.Sequential(nn.Linear(state_size, hidden_dim), self.norm, self.act, self.dropout)  # First layer embeds the input to hidden_dim

        # Hidden Layers
        for _ in range(layers):
            self.network.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), self.norm, self.act, self.dropout))

        # Output head
        self.output = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, x):
        """
        x: [batch_dim, state_dim]
        returns: (logits, value)
          logits: [batch_dim, action_dim]
          value:  [batch_dim, 1]
        """
        # Input embedding
        sc = self.embed(x)  # Initial embedding acts as the entire network's residual connection

        # Layers
        x = sc  # Start with the embedding
        for layer in self.network:
            x = (x + layer(x)) / sqrt(2)  # Residual within layers

        # Overall residual connection
        x = (x + sc) / sqrt(2)

        # Output
        reward = (1 - self.output(x)) * self.vocab_entropy  # shape [batch_dim, 1]

        return reward


# reward_net = RewardNet(state_size=26*11+6, hidden_dim=128, layers=3, dropout=0.1).to(device)
# reward_net.load_state_dict(torch.load("wordle/environment/reward_model/best_small_reward_net.pth", weights_only=True, map_location=device))
# reward_net.eval()