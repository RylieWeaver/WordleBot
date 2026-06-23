# General

# Torch

# Wordle
from .model import (
    ActorCriticNetConfig, ActorCriticNet,
    SeparatedActorCriticNetConfig, SeparatedActorCriticNet,
    DotGuessStateNetConfig, DotGuessStateNet,
    DotGuessStateNet2Config, DotGuessStateNet2,
    WordleTransformerConfig, WordleTransformer
)

"""
The purpose of this file is to allow the Trainer() to resume training from a checkpoint
regardless of which model was used. To do so, it needs to have a mapping of names to
model classes so that it can dynamically instantiate the correct model given a config.

NOTE: The model configs must set the same naming.
"""

name2model = {
    "ActorCriticNet": [ActorCriticNetConfig, ActorCriticNet],
    "SeparatedActorCriticNet": [SeparatedActorCriticNetConfig, SeparatedActorCriticNet],
    "DotGuessStateNet": [DotGuessStateNetConfig, DotGuessStateNet],
    "DotGuessStateNet2": [DotGuessStateNet2Config, DotGuessStateNet2],
    "WordleTransformer": [WordleTransformerConfig, WordleTransformer],
}
