# General
from typing import Dict, Any, Literal, Optional

# Torch
import torch
from torch.optim import Adam, AdamW, SGD, RMSprop, Adagrad, Adadelta, Rprop, LBFGS

# Wordle
from wordle.utils import Config, filtered_kwargs



STR2OPT = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
    "rmsprop": RMSprop,
    "adagrad": Adagrad,
    "adadelta": Adadelta,
    "rprop": Rprop,
    "lbfgs": LBFGS,
}

class OptHandlerConfig(Config):
    def __init__(
            self,
            name: Literal["adam","adamw","sgd","rmsprop","adagrad","adadelta","rprop","lbfgs"] = "adamw",
            lr: float = 1e-4,
            grad_clip: Optional[float] = 1.0,
            **kwargs: Optional[Dict[str, Any]]
        ):
        self.name = name
        self.lr = lr
        self.grad_clip = grad_clip
        self.extras = kwargs if kwargs else None

class OptHandler:
    def __init__(self, cfg: OptHandlerConfig):
        self.cfg = cfg
        self.name = self.cfg.name
        self.lr = self.cfg.lr
        self.grad_clip = self.cfg.grad_clip
        self.extras = self.cfg.extras

    def build_optimizer(self, params) -> torch.optim.Optimizer:
        opt_cls = STR2OPT[self.name]
        kwargs: Dict[str, Any] = {"lr": self.lr}
        if self.extras:
            kwargs.update(filtered_kwargs(opt_cls, self.extras))
        return opt_cls(params, **kwargs)
    
    def clip_grad_norm(self, model: torch.nn.Module):
        if (self.grad_clip is not None) and (self.grad_clip is not False):
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
