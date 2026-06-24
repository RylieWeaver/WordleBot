from typing import Tuple

import torch
import torch.nn.functional as F

from wordle.utils import Config, to_float


class EnumLossComponent:
    ACTOR = "actor"
    CRITIC = "critic"
    ENTROPY = "entropy"
    KL_REG = "kl_reg"
    KL_GUIDE = "kl_guide"


class WordleLossConfig(Config):
    def __init__(
            self,
            loss_weights: dict = None,
            ratio_prob_clip: float = 0.2,
            min_prob_clip: float = -0.01,
            max_prob_clip: float = 0.95,
            eps: float = 1e-12,
            clamp: float = 1e-12
    ):
        self.loss_weights = loss_weights if loss_weights is not None else self.default_weights()
        self.ratio_prob_clip = ratio_prob_clip
        self.min_prob_clip = min_prob_clip
        self.max_prob_clip = max_prob_clip
        self.eps = eps
        self.clamp = clamp

    def default_weights(self):
        return {
            EnumLossComponent.ACTOR: 1.0,
            EnumLossComponent.CRITIC: 0.0,
            EnumLossComponent.ENTROPY: 0.0,
            EnumLossComponent.KL_REG: 0.0,
            EnumLossComponent.KL_GUIDE: 0.0,
        }


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
        loss = 0.0
        loss_components = {
            EnumLossComponent.ACTOR: 0.0,
            EnumLossComponent.CRITIC: 0.0,
            EnumLossComponent.ENTROPY: 0.0,
            EnumLossComponent.KL_REG: 0.0,
            EnumLossComponent.KL_GUIDE: 0.0,
        }
        return loss, loss_components

    def init_cumulative_loss(self):
        self.loss, self.loss_components = self.make_loss()

    def measure_grad_norms(self, *args, **kwargs):
        print("Skip grad norms for now")

    def calculate_loss_components(self, states, actions, responses, probs, ref_probs=None) -> Tuple[torch.Tensor, ...]:
        active_mask = states["active_mask"][:, :-1, ...]
        target_probs = actions["search_target_probs"].detach()
        policy_probs = probs["policy_probs_masked"]
        ref_policy_probs = ref_probs["policy_probs_masked"][:, :-1, ...] if ref_probs is not None else None

        target_active = target_probs[active_mask]
        policy_active = policy_probs[active_mask]
        log_policy_active = self.log_normalize(policy_active)
        log_target_active = self.log_normalize(target_active)

        # Main search-distillation objective: KL(target_search || model_policy).
        actor_loss = F.kl_div(log_policy_active, target_active, reduction="batchmean", log_target=False)
        critic_loss = torch.zeros((), dtype=policy_active.dtype, device=policy_active.device)
        entropy_loss = torch.zeros((), dtype=policy_active.dtype, device=policy_active.device)
        if ref_policy_probs is not None:
            ref_active = ref_policy_probs[active_mask]
            kl_reg_loss = F.kl_div(self.log_normalize(policy_active), ref_active.detach(), reduction="batchmean", log_target=False)
        else:
            kl_reg_loss = torch.zeros((), dtype=policy_active.dtype, device=policy_active.device)
        kl_guide_loss = F.kl_div(log_policy_active, target_active, reduction="batchmean", log_target=False)
        return (actor_loss, critic_loss, entropy_loss, kl_reg_loss, kl_guide_loss)

    def inc_loss(self, states, actions, responses, probs, ref_probs=None):
        batch_loss, batch_loss_components = self.make_loss()
        losses = self.calculate_loss_components(states, actions, responses, probs, ref_probs)
        keys = [
            EnumLossComponent.ACTOR,
            EnumLossComponent.CRITIC,
            EnumLossComponent.ENTROPY,
            EnumLossComponent.KL_REG,
            EnumLossComponent.KL_GUIDE,
        ]
        for key, loss in zip(keys, losses):
            batch_loss_components[key] = to_float(loss)
            batch_loss = batch_loss + self.cfg.loss_weights.get(key, 0.0) * loss
        self.loss = self.loss + batch_loss.item()
        for k in self.loss_components.keys():
            self.loss_components[k] = self.loss_components[k] + batch_loss_components[k]
        return batch_loss, batch_loss_components

    def average_cumulative_loss(self, num_batches):
        self.loss = self.loss / num_batches
        for key in self.loss_components:
            self.loss_components[key] = self.loss_components[key] / num_batches

    def compute_metrics(self):
        metrics = {"loss": self.loss}
        metrics.update(self.loss_components)
        return metrics
