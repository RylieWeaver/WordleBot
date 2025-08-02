# General
from math import sqrt

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F



##############################################
# SC-BLOCK
##############################################
class MLPBlock(nn.Module):
    def __init__(self, dim, activation=nn.SiLU(), dropout=0.0):
        super().__init__()
        self.res = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            activation,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.res(x)



############################################
# ACTOR-CRITIC NETWORK
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
            self.network.append(MLPBlock(hidden_dim, self.act, dropout))

        # Output heads
        self.logits = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
        self.value = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        """
        x: [batch_size, *, state_dim]
        returns: (logits, value)
          logits: [batch_size, *, output_dim]
          value:  [batch_size, *, 1]
        """

        # Input embedding
        sc = self.embed(x)  # Initial embedding acts as the entire network's residual connection

        # Layers
        x = sc  # Start with the embedding
        for layer in self.network:
            x = layer(x)

        # Overall residual connection
        x = (x + sc)

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
        self.actor_embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), self.act, self.dropout)
        self.critic_embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), self.act, self.dropout)

        for _ in range(layers):
            self.actor_network.append(MLPBlock(hidden_dim, self.act, dropout))
            self.critic_network.append(MLPBlock(hidden_dim, self.act, dropout))

        # Output heads
        self.logits = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
        self.value = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        """
        x: [batch_size, *, state_dim]
        returns: (logits, value)
          logits: [batch_size, *, output_dim]
          value:  [batch_size, *, 1]
        """

        # Input embedding
        actor_sc = self.actor_embed(x)
        critic_sc = self.critic_embed(x)

        # Layers
        x1, x2 = actor_sc, critic_sc  # Start with the embedding
        for actor_layer, critic_layer in zip(self.actor_network, self.critic_network):
            x1 = actor_layer(x1)
            x2 = critic_layer(x2)

        # Overall residual connection
        x1 = x1 + actor_sc
        x2 = x2 + critic_sc

        # Output heads
        policy_logits = self.logits(x1)  # [batch_size, output_dim]
        state_value = self.value(x2)  # [batch_size, 1]

        return policy_logits, state_value



############################################
# DOT GUESS STATE NETWORK
############################################
class DotGuessStateNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, total_vocab_tensor, layers=3, dropout=0.1, device='cpu'):
        super().__init__()
        self.state_layers = nn.ModuleList()
        self.guess_layers = nn.ModuleList()
        self.hidden_dim = hidden_dim
        guess_dim = max(hidden_dim // 16, 16)
        self.guess_dim = guess_dim
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        output_dim = total_vocab_tensor.shape[0]
        self.register_buffer("guess_states", F.one_hot(total_vocab_tensor, num_classes=26).float().permute(0, 2, 1).reshape(output_dim, -1).to(torch.float32))  # [total_vocab_size, 26*5]

        # Embedding
        self.state_embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), self.act, self.dropout)
        self.guess_embed = nn.Sequential(nn.Linear(130, guess_dim), self.act, self.dropout)

        # Layers
        for _ in range(layers):
            self.state_layers.append(MLPBlock(hidden_dim, activation=self.act, dropout=dropout))
            self.guess_layers.append(MLPBlock(guess_dim, activation=self.act, dropout=dropout))

        # Guess state attention
        self.state_q = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.guess_k = nn.Sequential(nn.LayerNorm(guess_dim), nn.Linear(guess_dim, hidden_dim))

        # Output heads
        self.value = nn.Sequential(
            MLPBlock(hidden_dim, activation=self.act, dropout=dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize guess keys
        self.register_buffer("guess_k_static", torch.empty([self.vocab_size, hidden_dim]))   # will hold cached keys in eval

    def _build_guess_k(self):
        # Input embedding
        sc_g = self.guess_embed(self.guess_states)  # [total_vocab_size, hidden_dim]
        # Layers
        g = sc_g
        for guess_layer in self.guess_layers:
            g = guess_layer(g)
        g = (g + sc_g)  # [total_vocab_size, hidden_dim]
        # Key projection
        k = self.guess_k(g)  # [total_vocab_size, hidden_dim]
        return k

    def eval(self):
        super().eval()
        with torch.no_grad():
            self.guess_k_static = self._build_guess_k().detach()
        return self

    def forward(self, x):
        """
        x: [batch_size, *, state_dim]
        returns: (logits, value)
          logits: [batch_size, *, total_vocab_size]
          value:  [batch_size, *, 1]
        """
        # State embedding
        sc_x = self.state_embed(x)  # [batch_size, *, hidden_dim]
        # Layers
        x = sc_x
        for state_layer in self.state_layers:
            x = state_layer(x)
        x = (x + sc_x)  # [batch_size, *, hidden_dim]

        # Guess key embeddings
        k = self._build_guess_k() if self.training else self.guess_k_static

        # Logit
        q = self.state_q(x)  # [batch_size, *, hidden_dim]
        scores = (q @ k.T) / sqrt(self.hidden_dim)  # [batch_size, *, total_vocab_size]

        # Value
        state_value = self.value(x)  # [batch_size, *, 1]

        return scores, state_value



############################################
# DOT GUESS STATE NETWORK
############################################
class DotGuessStateNet2(nn.Module):
    def __init__(self, input_dim, hidden_dim, total_vocab_tensor, layers=3, dropout=0.1, device='cpu'):
        super().__init__()
        self.state_layers = nn.ModuleList()
        self.letter_layers = nn.ModuleList()
        self.hidden_dim = hidden_dim
        self.vocab_size = total_vocab_tensor.shape[0]
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        offsets = 26*torch.arange(5, device=device).unsqueeze(0)  # [1, 5]
        self.register_buffer("guess_idxs", (total_vocab_tensor + offsets).to(torch.long))  # [total_vocab_size, 5]
        self.register_buffer("letter_mask", torch.arange(130, device=device).to(torch.long))  # [130]

        # Embedding
        self.state_embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), self.act, self.dropout)
        self.letter_embed = nn.Sequential(nn.Embedding(130, hidden_dim), self.act, self.dropout)

        # Layers
        for _ in range(layers):
            self.state_layers.append(MLPBlock(hidden_dim, activation=self.act, dropout=dropout))
            self.letter_layers.append(MLPBlock(hidden_dim, activation=self.act, dropout=dropout))

        # Guess state attention
        self.state_q = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.letter_k = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))

        # Output heads
        self.value = nn.Sequential(
            MLPBlock(hidden_dim, activation=self.act, dropout=dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize guess keys
        self.register_buffer("guess_k_static", torch.empty([self.vocab_size, hidden_dim]))   # will hold cached keys in eval

    def _build_guess_k(self):
        # Input embedding
        sc_l = self.letter_embed(self.letter_mask)  # [130, hidden_dim]
        # Layers
        l = sc_l
        for letter_layer in self.letter_layers:
            l = letter_layer(l)
        l = (l + sc_l)  # [130, hidden_dim]
        # Key projection and mean
        l = self.guess_k(l)  # [130, hidden_dim]
        k = l[self.guess_idxs].mean(dim=-2)  # [total_vocab_size, hidden_dim]
        return k

    def eval(self):
        super().eval()
        with torch.no_grad():
            self.guess_k_static = self._build_guess_k().detach()
        return self

    def forward(self, x):
        """
        x: [batch_size, *, state_dim]
        returns: (logits, value)
          logits: [batch_size, *, total_vocab_size]
          value:  [batch_size, *, 1]
        """
        # State embedding
        sc_x = self.state_embed(x)  # [batch_size, *, hidden_dim]
        # Layers
        x = sc_x
        for state_layer in self.state_layers:
            x = state_layer(x)
        x = (x + sc_x)  # [batch_size, *, hidden_dim]

        # Guess key embeddings
        k = self._build_guess_k() if self.training else self.guess_k_static

        # Logit
        q = self.state_q(x)  # [batch_size, *, hidden_dim]
        scores = (q @ k.T) / sqrt(self.hidden_dim)  # [batch_size, *, total_vocab_size]

        # Value
        state_value = self.value(x)  # [batch_size, *, 1]

        return scores, state_value
