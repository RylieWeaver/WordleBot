# General
import math
from typing import Optional
from math import sqrt

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Wordle
from wordle.utils import Config
from .base import WordleModel, DotWordleModel
from .utils import MLPBlock, TransformerBlock



############################################
# TIME ONE-HOT ENCODING
############################################
class TimeOHE(nn.Module):
    def __init__(self, max_guesses: int):
        super().__init__()
        # row 0 = zeros, rows 1..G = one‑hots
        table = torch.zeros(max_guesses + 1, max_guesses, dtype=torch.float32)
        table[1:, :] = torch.eye(max_guesses, dtype=torch.float32)
        self.register_buffer("lu_table", table, persistent=False)

    def forward(self, t):
        # t can be any shape, assumes t starts at 1
        t = t.to(torch.long)
        valid = (t >= 1) & (t <= self.lu_table.size(0) - 1)
        idx = torch.where(valid, t, torch.zeros_like(t))
        return self.lu_table[idx]


def _linear_flops(input_dim, output_dim, bias=True, tokens=1):
    return tokens * ((2 * input_dim * output_dim) + (output_dim if bias else 0))


def _layernorm_flops(dim, tokens=1):
    return tokens * 2 * dim


def _mlp_block_flops(dim, tokens=1):
    return _layernorm_flops(dim, tokens=tokens) + _linear_flops(dim, dim, tokens=tokens)


def _ff_flops(dim, tokens=1):
    return _linear_flops(dim, 4 * dim, tokens=tokens) + _linear_flops(4 * dim, dim, tokens=tokens)


def _mha_flops(dim, bias=False, tokens=1):
    return _linear_flops(dim, 3 * dim, bias=bias, tokens=tokens)


def _transformer_block_flops(dim, bias=False, tokens=1):
    return (
        _layernorm_flops(dim, tokens=tokens) +
        _mha_flops(dim, bias=bias, tokens=tokens) +
        _layernorm_flops(dim, tokens=tokens) +
        _ff_flops(dim, tokens=tokens)
    )


############################################
# ACTOR-CRITIC NETWORK
############################################
class ActorCriticNetConfig(Config):
    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            num_letters: int = 26,
            letter_input_dim: int = 11,
            max_guesses: int = 6,
            layers: int = 3,
            dropout: float = 0.1,
            use_inductive_biases: bool = True,
        ):
        self.model_name = "ActorCriticNet"
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_letters = num_letters
        self.letter_input_dim = letter_input_dim
        self.max_guesses = max_guesses
        self.layers = layers
        self.dropout = dropout
        self.use_inductive_biases = use_inductive_biases

class ActorCriticNet(WordleModel):
    def __init__(self, cfg: ActorCriticNetConfig, device=None):
        super().__init__(use_inductive_biases=cfg.use_inductive_biases, device=device)
        self.network = nn.ModuleList()
        self.act = nn.SiLU()
        # Read (aliases for convenience)
        self.cfg = cfg
        self.hidden_dim = H = self.cfg.hidden_dim
        self.output_dim = O = self.cfg.output_dim
        self.num_letters = N = self.cfg.num_letters
        self.letter_input_dim = D = self.cfg.letter_input_dim
        self.max_guesses = G = self.cfg.max_guesses
        self.layers = L = self.cfg.layers
        self.dropout = self.cfg.dropout
        input_dim = I = (N * D) + G

        # Embedding
        self.t_ohe = TimeOHE(max_guesses=G)
        self.embed = nn.Sequential(nn.Linear(I, H), self.act, nn.Dropout(self.dropout), nn.LayerNorm(H))
        # Layers
        for _ in range(self.layers):
            self.network.append(MLPBlock(H, self.act, self.dropout))

        # Output
        self.logits = nn.Sequential(
            MLPBlock(H, activation=self.act, dropout=self.dropout),
            nn.LayerNorm(H),
            nn.Linear(H, O)
        )
        self.value = nn.Sequential(
            MLPBlock(H, activation=self.act, dropout=self.dropout),
            nn.LayerNorm(H),
            nn.Linear(H, 1)
        )

    def forward_flops(self, num_states: int = 1):
        I = (self.num_letters * self.letter_input_dim) + self.max_guesses
        H = self.hidden_dim
        return num_states * (
            _linear_flops(I, H) + _layernorm_flops(H) +
            self.layers * _mlp_block_flops(H) +
            _mlp_block_flops(H) + _layernorm_flops(H) + _linear_flops(H, self.output_dim) +
            _mlp_block_flops(H) + _layernorm_flops(H) + _linear_flops(H, 1)
        )

    def forward(self, states):
        # Unpack
        a = states["alphabet"].flatten(start_dim=-2).float()        # [B, *, 26, 11] --> [B, *, 26*11]
        t = self.t_ohe(states["t"])                                 # [B, *, G]
        x = torch.cat((a, t), dim=-1)                               # [B, *, 26*11 + G] = [B, *, input_dim]

        # Input embedding
        x = self.embed(x)  # Initial embedding acts as the entire network's residual connection

        # Layers
        for layer in self.network:
            x = layer(x)

        # Output heads
        policy_logits = self.logits(x)              # [B, *, V]
        state_value = self.value(x).squeeze(-1)     # [B, *]

        return policy_logits, state_value



############################################
#  SEPARATED  ACTOR–CRITIC  NETWORK
############################################
class SeparatedActorCriticNetConfig(Config):
    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            num_letters: int = 26,
            letter_input_dim: int = 11,
            max_guesses: int = 6,
            layers: int = 3,
            dropout: float = 0.1,
            use_inductive_biases: bool = True,
        ):
        self.model_name = "SeparatedActorCriticNet"
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_letters = num_letters
        self.letter_input_dim = letter_input_dim
        self.max_guesses = max_guesses
        self.layers = layers
        self.dropout = dropout
        self.use_inductive_biases = use_inductive_biases

class SeparatedActorCriticNet(WordleModel):
    def __init__(self, cfg: SeparatedActorCriticNetConfig, device=None):
        super().__init__(use_inductive_biases=cfg.use_inductive_biases, device=device)
        self.actor_network = nn.ModuleList()
        self.critic_network = nn.ModuleList()
        self.act = nn.SiLU()
        # Read (aliases for convenience)
        self.cfg = cfg
        self.hidden_dim = H = self.cfg.hidden_dim
        self.output_dim = O = self.cfg.output_dim
        self.num_letters = N = self.cfg.num_letters
        self.letter_input_dim = D = self.cfg.letter_input_dim
        self.max_guesses = G = self.cfg.max_guesses
        self.layers = L = self.cfg.layers
        self.dropout = self.cfg.dropout
        input_dim = I = (N * D) + G

        # Embedding
        self.t_ohe = TimeOHE(max_guesses=G)
        self.actor_embed = nn.Sequential(nn.Linear(I, H), self.act, nn.Dropout(self.dropout), nn.LayerNorm(H))
        self.critic_embed = nn.Sequential(nn.Linear(I, H), self.act, nn.Dropout(self.dropout), nn.LayerNorm(H))

        # Layers
        for _ in range(L):
            self.actor_network.append(MLPBlock(H, self.act, self.dropout))
            self.critic_network.append(MLPBlock(H, self.act, self.dropout))

        # Output
        self.logits = nn.Sequential(
            MLPBlock(H, activation=self.act, dropout=self.dropout),
            nn.LayerNorm(H),
            nn.Linear(H, O)
        )
        self.value = nn.Sequential(
            MLPBlock(H, activation=self.act, dropout=self.dropout),
            nn.LayerNorm(H),
            nn.Linear(H, 1)
        )

    def forward_flops(self, num_states: int = 1):
        I = (self.num_letters * self.letter_input_dim) + self.max_guesses
        H = self.hidden_dim
        return num_states * (
            2 * (_linear_flops(I, H) + _layernorm_flops(H)) +
            2 * self.layers * _mlp_block_flops(H) +
            _mlp_block_flops(H) + _layernorm_flops(H) + _linear_flops(H, self.output_dim) +
            _mlp_block_flops(H) + _layernorm_flops(H) + _linear_flops(H, 1)
        )
    
    def forward(self, states):
        # Unpack
        a = states["alphabet"].flatten(start_dim=-2).float()        # [B, *, 26, 11] --> [B, *, 26*11]
        t = self.t_ohe(states["t"])                                 # [B, *, G]
        x = torch.cat((a, t), dim=-1)                               # [B, *, 26*11 + G] = [B, *, input_dim]

        # Input embedding
        x1 = self.actor_embed(x)
        x2 = self.critic_embed(x)

        # Layers
        for actor_layer, critic_layer in zip(self.actor_network, self.critic_network):
            x1 = actor_layer(x1)
            x2 = critic_layer(x2)

        # Output heads
        policy_logits = self.logits(x1)                 # [B, *, V]
        state_value = self.value(x2).squeeze(-1)        # [B, *]

        return policy_logits, state_value



############################################
# DOT GUESS STATE NETWORK
############################################
class DotGuessStateNetConfig(Config):
    def __init__(
            self,
            state_hidden_dim: int,
            guess_hidden_dim: int,
            output_dim: int,
            vocab_size: int = (2315 + 10657),
            num_letters: int = 26,
            letter_input_dim: int = 11,
            max_guesses: int = 6,
            layers: int = 3,
            dropout: float = 0.1,
            use_inductive_biases: bool = True,
        ):
        self.model_name = "DotGuessStateNet"
        self.state_hidden_dim = state_hidden_dim
        self.guess_hidden_dim = guess_hidden_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.num_letters = num_letters
        self.letter_input_dim = letter_input_dim
        self.max_guesses = max_guesses
        self.layers = layers
        self.dropout = dropout
        self.use_inductive_biases = use_inductive_biases

class DotGuessStateNet(DotWordleModel):
    def __init__(
            self,
            cfg: DotGuessStateNetConfig,
            total_vocab_tensor: Optional[torch.Tensor] = None,
            device=None
        ):
        super().__init__(use_inductive_biases=cfg.use_inductive_biases, device=device)
        self.state_layers = nn.ModuleList()
        self.guess_layers = nn.ModuleList()
        self.act = nn.SiLU()
        # Read (aliases for convenience)
        self.cfg = cfg
        self.vocab_size = T =self.cfg.vocab_size
        self.state_hidden_dim = HS = self.cfg.state_hidden_dim
        self.guess_hidden_dim = HG = self.cfg.guess_hidden_dim
        self.output_dim = O = self.cfg.output_dim
        self.num_letters = L = self.cfg.num_letters
        self.letter_input_dim = IL = self.cfg.letter_input_dim
        self.max_guesses = G = self.cfg.max_guesses
        self.layers = self.cfg.layers
        self.dropout = self.cfg.dropout
        num_letter_positions = P = 5
        state_input_dim = IS = (L * IL) + G
        guess_input_dim = IG = (L * P)  # (One-hot of 26 letters * 5 letter possibilities)
        
        # Embedding
        self.t_ohe = TimeOHE(max_guesses=G)
        self.state_embed = nn.Sequential(nn.LayerNorm(IS), nn.Linear(IS, HS), self.act, nn.Dropout(self.dropout))
        self.guess_embed = nn.Sequential(nn.LayerNorm(IG), nn.Linear(IG, HG), self.act, nn.Dropout(self.dropout))

        # Layers
        for _ in range(self.layers):
            self.state_layers.append(MLPBlock(HS, activation=self.act, dropout=self.dropout))
            self.guess_layers.append(MLPBlock(HG, activation=self.act, dropout=self.dropout))

        # Guess state attention
        self.state_q = nn.Sequential(
            MLPBlock(HS, activation=self.act, dropout=self.dropout),
            nn.LayerNorm(HS),
            nn.Linear(HS, self.output_dim, bias=False)
        )
        self.guess_k = nn.Sequential(
            MLPBlock(HG, activation=self.act, dropout=self.dropout),
            nn.LayerNorm(HG),
            nn.Linear(HG, self.output_dim, bias=False)
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # Value
        self.value = nn.Sequential(
            nn.Linear(HS + IG, HS),
            MLPBlock(HS, activation=self.act, dropout=self.dropout),
            nn.LayerNorm(HS),
            nn.Linear(HS, 1)
        )

        # Buffers that are part of the model state
        self.register_buffer("total_vocab_tensor", torch.empty(T, P, dtype=torch.long), persistent=True)
        self.register_buffer("guess_states", torch.empty(T, IG, dtype=torch.float32), persistent=True)
        self.register_buffer("guess_k_static", torch.empty(T, O, dtype=torch.float32), persistent=False)
        # Initialize if provided. Otherwise will be loaded/constructed when loading state dict or the model
        # # NOTE: the model starts in train mode, so the keys will be built on the first forward pass or .eval() call
        if total_vocab_tensor is not None:
            self._set_vocab_buffers(total_vocab_tensor)
            self.refresh_static_cache()

    def forward_flops(self, num_states: int = 1):
        IS = (self.num_letters * self.letter_input_dim) + self.max_guesses
        IG = self.num_letters * 5
        HS = self.state_hidden_dim
        HG = self.guess_hidden_dim
        state_flops = (
            _layernorm_flops(IS) + _linear_flops(IS, HS) +
            self.layers * _mlp_block_flops(HS) +
            _mlp_block_flops(HS) + _layernorm_flops(HS) + _linear_flops(HS, self.output_dim, bias=False) +
            _linear_flops(HS + IG, HS) + _mlp_block_flops(HS) + _layernorm_flops(HS) + _linear_flops(HS, 1)
        )
        guess_flops = (
            _layernorm_flops(IG) + _linear_flops(IG, HG) +
            self.layers * _mlp_block_flops(HG) +
            _mlp_block_flops(HG) + _layernorm_flops(HG) + _linear_flops(HG, self.output_dim, bias=False)
        )
        cache_flops = self.vocab_size * guess_flops if self.training else 0
        return (num_states * state_flops) + cache_flops

    def _set_vocab_buffers(self, total_vocab_tensor: torch.Tensor):
        device = self.total_vocab_tensor.device
        tv = total_vocab_tensor.to(device=device, dtype=torch.long)         # [T, P]
        self.total_vocab_tensor.copy_(tv)                                   # [T, P]
        tv_one_hot = F.one_hot(
            tv, num_classes=self.num_letters
        ).to(torch.float32)                                                 # [T, P, N]
        self.guess_states.copy_(tv_one_hot.flatten(1))                      # [T, L*P]

    def _build_guess_k(self):
        # Input embedding
        g = self.guess_embed(self.guess_states)     # [T, HG]

        # Layers
        for guess_layer in self.guess_layers:
            g = guess_layer(g)                      # [T, HG]
        
        # Key projection
        k = self.guess_k(g)                         # [T, O]
        return k

    def forward(self, states):
        # Unpack
        a = states["alphabet"].flatten(start_dim=-2).float()        # [B, *, L, IL] --> [B, *, L*IL]
        t = self.t_ohe(states["t"])                                 # [B, *, 1] --> [B, *, G]
        x = torch.cat((a, t), dim=-1)                               # [B, *, L*IL + G] = [B, *, IS]

        # State embedding
        x = self.state_embed(x)                                     # [B, *, HS]

        # Layers
        for state_layer in self.state_layers:
            x = state_layer(x)                                      # [B, *, HS]

        # Guess key embeddings
        k = self._build_guess_k() if self.training else self.guess_k_static

        # Logit
        q = self.state_q(x)                                         # [B, *, O]
        # logit_scale = self.logit_scale.clamp(min=0.01, max=100.0)
        qk_scale = 1 / sqrt(self.output_dim)
        scores = (q @ k.T) * qk_scale               # [B, *, T]
        # q = F.normalize(q, dim=-1)
        # k = F.normalize(k, dim=-1)
        # scores = logit_scale * (q @ k.T)
        
        # Value
        h = self.guess_states[states["idx"]]                        # [B, *, IG]
        state_value = (
            self.value(torch.cat((x, h), dim=-1))
        ).squeeze(-1)                                               # [B, *]
        return scores, state_value



############################################
# DOT GUESS STATE NETWORK2
############################################
class DotGuessStateNet2Config(Config):
    def __init__(
            self,
            state_hidden_dim: int,
            guess_hidden_dim: int,
            output_dim: int,
            vocab_size: int = (2315 + 10657),
            num_letters: int = 26,
            letter_input_dim: int = 11,
            max_guesses: int = 6,
            layers: int = 3,
            dropout: float = 0.1,
            use_inductive_biases: bool = True,
        ):
        self.model_name = "DotGuessStateNet2"
        self.state_hidden_dim = state_hidden_dim
        self.guess_hidden_dim = guess_hidden_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.num_letters = num_letters
        self.letter_input_dim = letter_input_dim
        self.max_guesses = max_guesses
        self.layers = layers
        self.dropout = dropout
        self.letter_hidden_dim = guess_hidden_dim
        self.use_inductive_biases = use_inductive_biases

class DotGuessStateNet2(DotWordleModel):
    def __init__(
            self,
            cfg: DotGuessStateNet2Config,
            total_vocab_tensor: Optional[torch.Tensor] = None,
            device=None
        ):
        super().__init__(use_inductive_biases=cfg.use_inductive_biases, device=device)
        self.state_layers = nn.ModuleList()
        self.letter_layers = nn.ModuleList()
        self.act = nn.SiLU()
        # Read (aliases for convenience)
        self.cfg = cfg
        self.vocab_size = T =self.cfg.vocab_size
        self.state_hidden_dim = HS = self.cfg.state_hidden_dim
        self.letter_hidden_dim = HL = self.cfg.letter_hidden_dim
        self.output_dim = O = self.cfg.output_dim
        self.num_letters = L = self.cfg.num_letters
        self.letter_input_dim = IL = self.cfg.letter_input_dim
        self.max_guesses = G = self.cfg.max_guesses
        self.layers = self.cfg.layers
        self.dropout = self.cfg.dropout
        num_letter_positions = P = 5
        self.num_letter_positions = P
        state_input_dim = IS = (L * IL) + G
        self.letter_onehot_dim = OHE_L = (L * P)  # (One-hot of 26 letters * 5 letter possibilities)

        # Embedding
        self.t_ohe = TimeOHE(max_guesses=G)
        self.state_embed = nn.Sequential(nn.LayerNorm(IS), nn.Linear(IS, HS), self.act, nn.Dropout(self.dropout))
        self.letter_embed = nn.Embedding(OHE_L, HL)

        # Layers
        for _ in range(self.layers):
            self.state_layers.append(MLPBlock(HS, activation=self.act, dropout=self.dropout))
            self.letter_layers.append(MLPBlock(HL, activation=self.act, dropout=self.dropout))

        # Guess state attention
        self.state_q = nn.Sequential(
            MLPBlock(HS, activation=self.act, dropout=self.dropout),
            nn.LayerNorm(HS), 
            nn.Linear(HS, O)
        )
        self.letter_k = nn.Sequential(
            MLPBlock(HL, activation=self.act, dropout=self.dropout),
            nn.LayerNorm(HL), 
            nn.Linear(HL, O)
        )

        # Output heads
        self.value = nn.Sequential(
            MLPBlock(HS, activation=self.act, dropout=self.dropout),
            nn.LayerNorm(HS), 
            nn.Linear(HS, 1)
        )

        # Buffers that are part of the model state
        self.register_buffer("total_vocab_tensor", torch.empty(T, P, dtype=torch.long), persistent=True)
        self.register_buffer("guess_idxs", torch.empty(T, P, dtype=torch.long), persistent=True)
        self.register_buffer("letter_mask", torch.empty(OHE_L, dtype=torch.long), persistent=True)
        self.register_buffer("guess_k_static", torch.empty(T, O, dtype=torch.float32), persistent=False)
        # Initialize if provided. Otherwise will be loaded/constructed when loading state dict or the model
        # # NOTE: the model starts in train mode, so the keys will be built on the first forward pass or .eval() call
        if total_vocab_tensor is not None:
            self._set_vocab_buffers(total_vocab_tensor)
            self.refresh_static_cache()

    def forward_flops(self, num_states: int = 1):
        IS = (self.num_letters * self.letter_input_dim) + self.max_guesses
        OHE_L = self.num_letters * 5
        HS = self.state_hidden_dim
        HL = self.letter_hidden_dim
        state_flops = (
            _layernorm_flops(IS) + _linear_flops(IS, HS) +
            self.layers * _mlp_block_flops(HS) +
            _mlp_block_flops(HS) + _layernorm_flops(HS) + _linear_flops(HS, self.output_dim) +
            _mlp_block_flops(HS) + _layernorm_flops(HS) + _linear_flops(HS, 1)
        )
        letter_flops = (
            self.layers * _mlp_block_flops(HL, tokens=OHE_L) +
            _mlp_block_flops(HL, tokens=OHE_L) +
            _layernorm_flops(HL, tokens=OHE_L) +
            _linear_flops(HL, self.output_dim, tokens=OHE_L)
        )
        cache_flops = letter_flops if self.training else 0
        return (num_states * state_flops) + cache_flops

    def _set_vocab_buffers(self, total_vocab_tensor: torch.Tensor):
        device = self.total_vocab_tensor.device
        tv = total_vocab_tensor.to(device=device, dtype=torch.long)                 # [T, P]
        self.total_vocab_tensor.copy_(tv)                                           # [T, P]
        offsets = self.num_letters*torch.arange(
            self.num_letter_positions, device=device
        ).unsqueeze(0)                                                              # [1, P]
        self.guess_idxs.copy_(tv + offsets)                                         # [T, P]
        self.letter_mask.copy_(
            torch.arange(self.letter_onehot_dim, device=device).to(torch.long)
        )                                                                           # [OHE_L]

    def _build_guess_k(self):
        # Input embedding
        l = self.letter_embed(self.letter_mask)     # [OHE_L] --> [HL]

        # Layers
        for letter_layer in self.letter_layers:
            l = letter_layer(l)                     # [HL]
        
        # Key projection and mean
        l = self.letter_k(l)                        # [OHE_L, O]
        k = l[self.guess_idxs].mean(dim=-2)         # [T, P, O] --> [T, O]
        return k

    def forward(self, states):
        # Unpack
        a = states["alphabet"].flatten(start_dim=-2).float()    # [B, * , L, D] --> [B, *, 26*11]
        t = self.t_ohe(states["t"])                             # [B, *, 1] --> [B, *, G]
        x = torch.cat((a, t), dim=-1)                           # [B, *, L*IL + G] = [B, *, state_input_dim]
        
        # State embedding
        x = self.state_embed(x)                                 # [B, *, HS]

        # Layers
        for state_layer in self.state_layers:
            x = state_layer(x)                                  # [B, *, HS]

        # Guess key embeddings
        k = self._build_guess_k() if self.training else self.guess_k_static     # [V, O]

        # Logit
        q = self.state_q(x)                                     # [B, *, O]
        scores = (q @ k.T) / sqrt(self.output_dim)              # [B, *, V]

        # Value
        state_value = self.value(x).squeeze(-1)                 # [B, *]
        return scores, state_value



############################################
# Wordle Transformer
############################################
class WordleTransformerConfig(Config):
    def __init__(
            self,
            state_hidden_dim: int,
            guess_hidden_dim: int,
            output_dim: int,
            vocab_size: int = (2315 + 10657),
            num_letters: int = 26,
            letter_input_dim: int = 11,
            max_guesses: int = 6,
            layers: int = 3,
            dropout: float = 0.1,
            use_inductive_biases: bool = True,
        ):
        self.model_name = "WordleTransformer"
        self.state_hidden_dim = state_hidden_dim
        self.guess_hidden_dim = guess_hidden_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.num_letters = num_letters
        self.letter_input_dim = letter_input_dim
        self.max_guesses = max_guesses
        self.layers = layers
        self.dropout = dropout
        self.use_inductive_biases = use_inductive_biases

class WordleTransformer(DotWordleModel):
    def __init__(
            self,
            cfg: WordleTransformerConfig,
            total_vocab_tensor: Optional[torch.Tensor] = None,
            device=None
        ):
        super().__init__(use_inductive_biases=cfg.use_inductive_biases, device=device)
        self.state_layers = nn.ModuleList()
        self.guess_layers = nn.ModuleList()
        self.act = nn.SiLU()
        # Read (aliases for convenience)
        self.cfg = cfg
        self.vocab_size = T =self.cfg.vocab_size
        self.state_hidden_dim = HS = self.cfg.state_hidden_dim
        self.guess_hidden_dim = HG = self.cfg.guess_hidden_dim
        self.output_dim = O = self.cfg.output_dim
        self.num_letters = L = self.cfg.num_letters
        self.letter_input_dim = IL = self.cfg.letter_input_dim
        self.max_guesses = G = self.cfg.max_guesses
        self.layers = self.cfg.layers
        self.dropout = self.cfg.dropout
        num_letter_positions = P = 5
        state_input_dim = IS = (L * IL) + G
        guess_input_dim = IG = (L * P)  # (One-hot of 26 letters * 5 letter possibilities)
        
        # Embedding
        self.t_ohe = TimeOHE(max_guesses=G)
        self.state_embed = nn.Sequential(nn.Linear(IS, HS), nn.LayerNorm(HS))
        self.guess_embed = nn.Sequential(nn.Linear(IG, HG), nn.LayerNorm(HG))

        # Layers
        for _ in range(self.layers):
            self.state_layers.append(TransformerBlock(HS, activation=self.act, dropout=self.dropout))
            self.guess_layers.append(TransformerBlock(HG, activation=self.act, dropout=self.dropout))

        # Guess state attention
        self.state_q = nn.Sequential(
            MLPBlock(HS, activation=self.act, dropout=self.dropout),
            nn.LayerNorm(HS),
            nn.Linear(HS, O),
            nn.LayerNorm(O)
        )
        self.guess_k = nn.Sequential(
            MLPBlock(HG, activation=self.act, dropout=self.dropout),
            nn.LayerNorm(HG),
            nn.Linear(HG, O),
            nn.LayerNorm(O)
        )

        # Output
        self.value = nn.Sequential(
            MLPBlock(HS, activation=self.act, dropout=self.dropout),
            nn.LayerNorm(HS),
            nn.Linear(HS, 1)
        )

        # Buffers that are part of the model state
        self.register_buffer("total_vocab_tensor", torch.empty(T, P, dtype=torch.long), persistent=True)
        self.register_buffer("guess_states", torch.empty(T, IG, dtype=torch.float32), persistent=True)
        self.register_buffer("guess_k_static", torch.empty(T, O, dtype=torch.float32), persistent=False)
        # Initialize if provided. Otherwise will be loaded/constructed when loading state dict or the model
        # # NOTE: the model starts in train mode, so the keys will be built on the first forward pass or .eval() call
        if total_vocab_tensor is not None:
            self._set_vocab_buffers(total_vocab_tensor)
            self.refresh_static_cache()

    def forward_flops(self, num_states: int = 1):
        IS = (self.num_letters * self.letter_input_dim) + self.max_guesses
        IG = self.num_letters * 5
        HS = self.state_hidden_dim
        HG = self.guess_hidden_dim
        state_tokens = 1  # Current implementation flattens alphabet state before state_embed.
        guess_tokens = 1  # Current implementation flattens each candidate word before guess_embed.
        state_flops = (
            _linear_flops(IS, HS, tokens=state_tokens) + _layernorm_flops(HS, tokens=state_tokens) +
            self.layers * _transformer_block_flops(HS, tokens=state_tokens) +
            _mlp_block_flops(HS, tokens=state_tokens) +
            _layernorm_flops(HS, tokens=state_tokens) +
            _linear_flops(HS, self.output_dim, tokens=state_tokens) +
            _layernorm_flops(self.output_dim, tokens=state_tokens) +
            _mlp_block_flops(HS, tokens=state_tokens) +
            _layernorm_flops(HS, tokens=state_tokens) +
            _linear_flops(HS, 1, tokens=state_tokens)
        )
        guess_flops = (
            _linear_flops(IG, HG, tokens=guess_tokens) + _layernorm_flops(HG, tokens=guess_tokens) +
            self.layers * _transformer_block_flops(HG, tokens=guess_tokens) +
            _mlp_block_flops(HG, tokens=guess_tokens) +
            _layernorm_flops(HG, tokens=guess_tokens) +
            _linear_flops(HG, self.output_dim, tokens=guess_tokens) +
            _layernorm_flops(self.output_dim, tokens=guess_tokens)
        )
        cache_flops = self.vocab_size * guess_flops if self.training else 0
        return (num_states * state_flops) + cache_flops

    def _set_vocab_buffers(self, total_vocab_tensor: torch.Tensor):
        device = self.total_vocab_tensor.device
        tv = total_vocab_tensor.to(device=device, dtype=torch.long)         # [T, P]
        self.total_vocab_tensor.copy_(tv)                                   # [T, P]
        tv_one_hot = F.one_hot(
            tv, num_classes=self.num_letters
        ).to(torch.float32)                                                 # [T, P, N]
        self.guess_states.copy_(tv_one_hot.flatten(1))                      # [T, L*P]

    def _build_guess_k(self):
        # Input embedding
        g = self.guess_embed(self.guess_states).unsqueeze(-2)     # [T, 1, HG]

        # Layers
        for guess_layer in self.guess_layers:
            g = guess_layer(g)                                      # [T, 1, HG]
        g = g.squeeze(-2)                                           # [T, HG]
        
        # Key projection
        k = self.guess_k(g)                         # [T, O]
        return k

    def forward(self, states):
        # Unpack
        a = states["alphabet"].flatten(start_dim=-2).float()        # [B, *, L, IL] --> [B, *, L*IL]
        t = self.t_ohe(states["t"])                                 # [B, *, G]
        x = torch.cat((a, t), dim=-1)                               # [B, *, L*IL + G] = [B, *, IS]

        # State embedding
        x = self.state_embed(x).unsqueeze(-2)                       # [B, *, 1, HS]

        # Layers
        for state_layer in self.state_layers:
            x = state_layer(x)                                      # [B, *, 1, HS]
        x = x.squeeze(-2)                                           # [B, *, HS]

        # Guess key embeddings
        k = self._build_guess_k() if self.training else self.guess_k_static

        # Logit
        q = self.state_q(x)                                         # [B, *, O]
        scores = (q @ k.T) / sqrt(self.output_dim)                  # [B, *, T]

        # Value
        state_value = self.value(x).squeeze(-1)                     # [B, *]
        return scores, state_value
