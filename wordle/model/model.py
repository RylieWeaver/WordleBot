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
    def __init__(self, dim, activation=nn.SiLU(), dropout=0.1):
        super().__init__()
        self.res = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            activation,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return (x + self.res(x)) / sqrt(2)



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
# GUESS STATE NETWORK
############################################
class GuessStateNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, total_vocab_tensor, layers=3, dropout=0.1, device='cpu'):
        super().__init__()
        self.pre_layers = nn.ModuleList()
        self.filter = nn.ModuleList()
        self.post_layers = nn.ModuleList()
        self.hidden_dim = hidden_dim
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        output_dim = total_vocab_tensor.shape[0]
        self.register_buffer("guess_states", F.one_hot(total_vocab_tensor, num_classes=26).float().permute(0, 2, 1).reshape(output_dim, -1).to(torch.float32))  # [total_vocab_size, 26*5]

        # Embed layer
        self.state_embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), self.act, self.dropout)
        self.guess_embed = nn.Sequential(nn.Linear(130, hidden_dim), self.act, self.dropout)

        # Pre-filter layers
        for _ in range(layers):
            # self.pre_layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout))
            self.pre_layers.append(MLPBlock(hidden_dim, activation=self.act, dropout=dropout))

        # Filter layers via guess state attention
        self.state_q = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.guess_k = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.scores_down = nn.Sequential(self.dropout, nn.Linear(output_dim, hidden_dim), self.act, self.dropout)
        self.filter_combine = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), self.act, self.dropout)

        # Post-filter layers
        for _ in range(layers):
            # self.post_layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout))
            self.post_layers.append(MLPBlock(hidden_dim, activation=self.act, dropout=dropout))
        
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
        sc_pre = self.state_embed(x)  # Initial embedding acts as the entire network's residual connection
        g = self.guess_embed(self.guess_states)  # [total_vocab_size, hidden_dim]

        # Pre-filter layers
        x_pre = sc_pre  # Start with the embedding
        for layer in self.pre_layers:
            x_pre = layer(x_pre)  # Residual within layers
        # Overall residual connection
        x_pre = (x_pre + sc_pre)  # [batch_size, *, hidden_dim]

        # Filter
        q = self.state_q(x_pre)  # [batch_size, *, hidden_dim]
        k = self.guess_k(g)  # [total_vocab_size, hidden_dim]
        scores = (q @ k.T) / sqrt(self.hidden_dim)  # [batch_size, *, total_vocab_size]
        scores = self.scores_down(scores)
        x_post = self.filter_combine(torch.cat([x_pre, scores], dim=-1))  # [batch_size, *, hidden_dim]
        
        # Post filter layers
        sc_post = x_post  # Start with the embedding
        for layer in self.post_layers:
            x_post = layer(x_post)  # Residual within layers
        # Overall residual connection
        x_post = (x_post + sc_post)  # [batch_size, *, hidden_dim]

        # Output heads
        state_value = self.value(x_pre)  # [batch_size, *, 1]
        policy_logits = self.logit(x_post)  # [batch_size, *, total_vocab_size]

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
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        output_dim = total_vocab_tensor.shape[0]
        self.register_buffer("guess_states", F.one_hot(total_vocab_tensor, num_classes=26).float().permute(0, 2, 1).reshape(output_dim, -1).to(torch.float32))  # [total_vocab_size, 26*5]

        # Embedding
        # self.state_embed = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), self.act, self.dropout)
        # self.guess_embed = nn.Sequential(nn.LayerNorm(130), nn.Linear(130, hidden_dim), self.act, self.dropout)
        self.state_embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), self.act, self.dropout)
        self.guess_embed = nn.Sequential(nn.Linear(130, hidden_dim), self.act, self.dropout)

        # Layers
        for _ in range(layers):
            self.state_layers.append(MLPBlock(hidden_dim, activation=self.act, dropout=dropout))
            self.guess_layers.append(MLPBlock(hidden_dim, activation=self.act, dropout=dropout))

        # Guess state attention
        self.state_q = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.guess_k = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        
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
        sc_x = self.state_embed(x)  # Initial embedding acts as the entire network's residual connection
        sc_g = self.guess_embed(self.guess_states)  # [total_vocab_size, hidden_dim]

        # Pre-filter layers
        x = sc_x
        g = sc_g
        for state_layer, guess_layer in zip(self.state_layers, self.guess_layers):
            x = state_layer(x)
            g = guess_layer(g)
        # Overall residual connection
        x = (x + sc_x) / sqrt(2)  # [batch_size, *, hidden_dim]
        g = (g + sc_g) / sqrt(2)  # [total_vocab_size, hidden_dim]

        # Logit
        q = self.state_q(x)  # [batch_size, *, hidden_dim]
        k = self.guess_k(g)  # [total_vocab_size, hidden_dim]
        scores = (q @ k.T) / sqrt(self.hidden_dim)  # [batch_size, *, total_vocab_size]

        # Value
        state_value = self.value(x)  # [batch_size, *, 1]

        return scores, state_value
    

    
############################################
# TRANSFORMER GUESS STATE NETWORK
############################################
class TransformerGuessStateNet(nn.Module):
    """
    • 26 letters ⇒ 26 token positions per timestep
    • each token has 11-dimentional state features
    • guess index ↔ positional embedding
    • multi-head attention stacks over tokens
    """
    def __init__(self, input_dim, hidden_dim, total_vocab_tensor, max_guess=6, layers=3, num_heads=4, dropout=0.1, device="cpu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.token_embed = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Linear(max_guess, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # Transformer encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4*hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # Filter guess states for prediction
        output_dim = total_vocab_tensor.shape[0]
        self.register_buffer("guess_filter", F.one_hot(total_vocab_tensor, num_classes=26).float().permute(0, 2, 1).reshape(output_dim, -1).to(torch.float32))  # [total_vocab_size, 26*5]
        self.filter_in = nn.Sequential(nn.Linear(hidden_dim, 130), nn.LayerNorm(130), self.act, self.dropout)
        self.filter_out = nn.Sequential(nn.Linear(output_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout)
        self.filter_combine = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), self.act, self.dropout)

        # Output heads
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head  = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: [batch_size, *, state_dim]
        returns: (logits, value)
          logits: [batch_size, *, total_vocab_size]
          value:  [batch_size, *, 1]
        """
        # Extract parts of state
        batch_size, *extra_dims, state_dim = x.shape
        alphabet_states = x[..., :-6].reshape(batch_size, *extra_dims, 26, 11)
        guess_states = x[..., -6:]

        # Embed state
        token_emb = self.token_embed(alphabet_states)
        pos_emb = self.pos_embed(guess_states).unsqueeze(-2)
        x_pre = self.dropout(token_emb + pos_emb)

        # Run transformer
        x_pre = x_pre.reshape(-1, 26, self.hidden_dim)  # Reshape to [batch_size, *, 26, hidden_dim]
        x_pre = self.transformer(x_pre)
        x_pre = x_pre.reshape(batch_size, *extra_dims, 26, self.hidden_dim)
        x_pre = x_pre.mean(dim=-2)

        # Filter with vocab states
        x_post = self.filter_in(x_pre)
        x_post = torch.matmul(x_post, self.guess_filter.T) / sqrt(self.hidden_dim)  # [batch_size, *, total_vocab_size]
        x_post = self.filter_out(x_post)
        x_post = self.filter_combine(torch.cat([x_pre, x_post], dim=-1))

        # Output heads
        policy_logits = self.policy_head(x_post)
        state_value = self.value_head(x_pre)

        return policy_logits, state_value
