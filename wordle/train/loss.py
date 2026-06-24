from typing import Tuple

import torch
import torch.nn.functional as F

from wordle.utils import Config, to_float


class EnumLossComponent:
    SEARCH_KL = "search_kl"


class WordleLossConfig(Config):
    def __init__(
            self,
            loss_weight: float = 1.0,
            eps: float = 1e-12,
            clamp: float = 1e-12,
    ):
        self.loss_weight = loss_weight
        self.eps = eps
        self.clamp = clamp


class WordleLoss:
    def __init__(self, cfg: WordleLossConfig):
        self.cfg = cfg
        self.eps = self.cfg.eps
        self.clamp = self.cfg.clamp
        self.init_cumulative_loss()

    def log_normalize(self, probs):
        probs = probs + self.eps
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.clamp)
        return torch.log(probs)

    def make_loss(self):
        return 0.0, {EnumLossComponent.SEARCH_KL: 0.0}

    def init_cumulative_loss(self):
        self.loss, self.loss_components = self.make_loss()

    def measure_grad_norms(self, *args, **kwargs):
        print("Skip grad norms for now")

    def calculate_loss_components(self, states, actions, responses, probs) -> Tuple[torch.Tensor]:
        active_mask = states["active_mask"][:, :-1, ...]
        target_probs = actions["search_target_probs"].detach()
        policy_probs = probs["policy_probs_masked"]

        target_active = target_probs[active_mask]
        policy_active = policy_probs[active_mask]
        log_policy_active = self.log_normalize(policy_active)

        # Main search-distillation objective: KL(target_search || model_policy).
        search_kl = F.kl_div(log_policy_active, target_active, reduction="batchmean", log_target=False)
        return (search_kl,)

    def inc_loss(self, states, actions, responses, probs):
        batch_loss, batch_loss_components = self.make_loss()
        (search_kl,) = self.calculate_loss_components(states, actions, responses, probs)
        batch_loss_components[EnumLossComponent.SEARCH_KL] = to_float(search_kl)
        batch_loss = batch_loss + self.cfg.loss_weight * search_kl

        self.loss = self.loss + batch_loss.item()
        self.loss_components[EnumLossComponent.SEARCH_KL] += batch_loss_components[EnumLossComponent.SEARCH_KL]
        return batch_loss, batch_loss_components

    def average_cumulative_loss(self, num_batches):
        self.loss = self.loss / num_batches
        for key in self.loss_components:
            self.loss_components[key] = self.loss_components[key] / num_batches

    def compute_metrics(self):
        metrics = {"loss": self.loss}
        metrics.update(self.loss_components)
        return metrics
