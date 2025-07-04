# General
from math import sqrt

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F


############################################
# NETWORK
############################################
class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=3, dropout=0.1, device='cpu'):
        super().__init__()
        self.network = nn.ModuleList()
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Embedding
        self.embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout)  # First layer embeds the input to hidden_dim

        # Layers
        for _ in range(layers):
            self.network.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout))

        # Output heads
        self.logits = nn.Linear(hidden_dim, output_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: [batch_size, state_dim]
        returns: (logits, value)
          logits: [batch_size, *, output_dim]
          value:  [batch_size, *, 1]
        """

        # Input embedding
        sc = self.embed(x)  # Initial embedding acts as the entire network's residual connection

        # Layers
        x = sc  # Start with the embedding
        for layer in self.network:
            x = (x + layer(x)) / sqrt(2)  # Residual within layers

        # Overall residual connection
        x = (x + sc) / sqrt(2)

        # Output heads
        policy_logits = self.logits(x)  # [batch_size, *, output_dim]
        state_value = self.value(x)  # [batch_size, *, 1]

        return policy_logits, state_value



############################################
#  SEPARATED  ACTOR–CRITIC  NETWORK
############################################
class SeparatedActorCriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=3, dropout=0.1, device='cpu'):
        super().__init__()
        self.actor_network = nn.ModuleList()
        self.critic_network = nn.ModuleList()
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Embedding layers
        self.actor_embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout)
        self.critic_embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout)

        for _ in range(layers):
            self.actor_network.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout))
            self.critic_network.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout))

        # Output heads
        self.logits = nn.Linear(hidden_dim, output_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: [batch_size, state_dim]
        returns: (logits, value)
          logits: [batch_size, output_dim]
          value:  [batch_size, 1]
        """

        # Input embedding
        actor_sc = self.actor_embed(x)
        critic_sc = self.critic_embed(x)

        # Layers
        x1, x2 = actor_sc, critic_sc  # Start with the embedding
        for actor_layer, critic_layer in zip(self.actor_network, self.critic_network):
            x1 = x1 + actor_layer(x1)
            x2 = x2 + critic_layer(x2)

        # Overall residual connection
        x1 = x1 + actor_sc
        x2 = x2 + critic_sc

        # Output heads
        policy_logits = self.logits(x1)  # [batch_size, output_dim]
        state_value = self.value(x2)  # [batch_size, 1]

        return policy_logits, state_value



############################################
# NEW GUESS STATE NETWORK
############################################
class GuessStateNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, total_vocab_tensor, layers=3, dropout=0.1, device='cpu'):
        super().__init__()
        self.pre_layers = nn.ModuleList()
        self.filter = nn.ModuleList()
        self.post_layers = nn.ModuleList()
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        output_dim = total_vocab_tensor.shape[0]
        self.register_buffer("guess_filter", F.one_hot(total_vocab_tensor, num_classes=26).float().permute(0, 2, 1).reshape(total_vocab_tensor.shape[0], -1).to(torch.float32))  # [total_vocab_size, 26*5]

        # Embed layers
        self.embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout)  # First layer embeds the input to hidden_dim

        # Pre-filter layers
        for _ in range(layers):
            self.pre_layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout))

        # Filter layers
        self.filter_in = nn.Sequential(nn.Linear(hidden_dim, 130), nn.LayerNorm(130), self.act, self.dropout)
        self.filter_out = nn.Sequential(nn.Linear(output_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout)
        self.filter_combine = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout)

        # Post-filter layers
        for _ in range(layers):
            self.post_layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout))
        
        # Output heads
        self.logit = nn.Linear(hidden_dim, output_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: [batch_size, *, state_dim]
        returns: (logits, value)
          logits: [batch_size, *, total_vocab_size]
          value:  [batch_size, *, 1]
        """
        # Input embedding
        sc_pre = self.embed(x)  # Initial embedding acts as the entire network's residual connection

        # Pre-filter layers
        x_pre = sc_pre  # Start with the embedding
        for layer in self.pre_layers:
            x_pre = (x_pre + layer(x_pre)) / sqrt(2)  # Residual within layers
        # Overall residual connection
        x_pre = (x_pre + sc_pre) / sqrt(2)  # [batch_size, *, hidden_dim]

        # Filter
        x_filtered = self.filter_in(x_pre)
        x_filtered = torch.matmul(x_filtered, self.guess_filter.T)
        x_filtered = self.filter_out(x_filtered)
        x_post = self.filter_combine(torch.cat((x_pre, x_filtered), dim=-1))

        # Post filter layers
        sc_post = x_post  # Start with the embedding
        for layer in self.post_layers:
            x_post = (x_post + layer(x_post)) / sqrt(2)  # Residual within layers
        # Overall residual connection
        x_post = (x_post + sc_post) / sqrt(2)  # [batch_size, *, hidden_dim]

        # Output heads
        state_value = self.value(x_pre)  # [batch_size, *, 1]
        policy_logits = self.logit(x_post)  # [batch_size, *, total_vocab_size]

        return policy_logits, state_value
