import torch
import torch.nn as nn
import torch.nn.functional as F


############################################
# NETWORK
############################################
class ActorCriticNet(nn.Module):
    def __init__(self, state_size, vocab_size, hidden_dim=128, layers=3, dropout=0.1):
        super().__init__()
        self.network = nn.ModuleList()
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Embedding layers
        self.embed = nn.Sequential(nn.Linear(state_size, hidden_dim), self.norm, self.act, self.dropout)  # First layer embeds the input to hidden_dim

        for _ in range(layers):
            self.network.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), self.norm, self.act, self.dropout))

        # Output heads
        self.logits = nn.Linear(hidden_dim, vocab_size)
        self.value = nn.Linear(hidden_dim, 1)

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
            x = x + layer(x)  # Residual within layers

        # Overall residual connection
        x = x + sc

        # Output heads
        policy_logits = self.logits(x)  # shape [batch_dim, action_dim]
        state_value = self.value(x)  # shape [batch_dim, 1]

        return policy_logits, state_value
