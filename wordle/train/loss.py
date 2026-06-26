from typing import Tuple

import torch
import torch.nn.functional as F

from wordle.utils import Config, to_float


class EnumLossComponent:
    SEARCH_KL = "search_kl"
    ENTROPY = "entropy"
    KL_REG = "kl_reg"
    KL_GUIDE = "kl_guide"
    SEARCH_COSINE = "search_cosine"
    UNIQUE_STATE_PCT = "unique_state_pct"


class WordleLossConfig(Config):
    def __init__(
            self,
            loss_weights: dict = None,
            eps: float = 1e-12,
            clamp: float = 1e-12,
    ):
        self.loss_weights = loss_weights if loss_weights is not None else self.default_weights()
        self.eps = eps
        self.clamp = clamp

    def default_weights(self):
        return {
            EnumLossComponent.SEARCH_KL: 1.0,
            EnumLossComponent.ENTROPY: 0.0,
            EnumLossComponent.KL_REG: 1.0,
            EnumLossComponent.KL_GUIDE: 0.25,
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
        return 0.0, {
            EnumLossComponent.SEARCH_KL: 0.0,
            EnumLossComponent.ENTROPY: 0.0,
            EnumLossComponent.KL_REG: 0.0,
            EnumLossComponent.KL_GUIDE: 0.0,
            EnumLossComponent.SEARCH_COSINE: 0.0,
            EnumLossComponent.UNIQUE_STATE_PCT: 0.0,
        }

    def init_cumulative_loss(self):
        self.loss, self.loss_components = self.make_loss()

    def measure_grad_norms(self, *args, **kwargs):
        print("Skip grad norms for now")

    def calculate_loss_components(self, states, actions, responses, probs, ref_probs=None) -> Tuple[torch.Tensor, ...]:
        active_mask = states["active_mask"][:, :-1, ...]
        target_probs = responses["search_target_probs"].detach()
        valid_mask = actions["valid_mask"]
        policy_probs = probs["policy_probs"]
        policy_probs_masked = probs["policy_probs_masked"]

        target_active = target_probs[active_mask]
        policy_active = policy_probs[active_mask]
        masked_active = policy_probs_masked[active_mask]
        valid_active = valid_mask[active_mask]
        log_policy_active = self.log_normalize(policy_active)
        log_masked_active = self.log_normalize(masked_active)

        # Main search-distillation objective: KL(target_search || masked model policy).
        search_kl = F.kl_div(log_masked_active, target_active, reduction="batchmean", log_target=False)

        # Entropy regularization over valid actions only. Minimizing negative entropy maximizes entropy.
        entropy_probs = policy_active * valid_active.float()
        entropy_probs = entropy_probs / entropy_probs.sum(dim=-1, keepdim=True).clamp_min(self.clamp)
        entropy_loss = torch.sum(entropy_probs * self.log_normalize(entropy_probs), dim=-1).mean()

        if ref_probs is not None:
            ref_active = ref_probs["policy_probs"][:, :-1, ...][active_mask].detach()
            kl_reg_loss = F.kl_div(log_policy_active, ref_active, reduction="batchmean", log_target=False)
        else:
            kl_reg_loss = torch.zeros((), dtype=policy_active.dtype, device=policy_active.device)

        # Guide the raw policy toward its valid-action-masked version so inductive-bias masking
        # becomes less necessary over time.
        kl_guide_loss = F.kl_div(log_policy_active, masked_active.detach(), reduction="batchmean", log_target=False)
        search_cosine = F.cosine_similarity(masked_active.detach(), target_active.detach(), dim=-1).mean()
        return (search_kl, entropy_loss, kl_reg_loss, kl_guide_loss, search_cosine)

    def inc_loss(self, states, actions, responses, probs, ref_probs=None):
        batch_loss, batch_loss_components = self.make_loss()
        losses = self.calculate_loss_components(states, actions, responses, probs, ref_probs)
        loss_keys = [
            EnumLossComponent.SEARCH_KL,
            EnumLossComponent.ENTROPY,
            EnumLossComponent.KL_REG,
            EnumLossComponent.KL_GUIDE,
        ]
        for key, loss in zip(loss_keys, losses[:len(loss_keys)]):
            batch_loss_components[key] = to_float(loss)
            batch_loss = batch_loss + self.cfg.loss_weights.get(key, 0.0) * loss
        batch_loss_components[EnumLossComponent.SEARCH_COSINE] = to_float(losses[-1])
        if "unique_state_pct" in responses:
            batch_loss_components[EnumLossComponent.UNIQUE_STATE_PCT] = to_float(responses["unique_state_pct"])

        self.loss = self.loss + batch_loss.item()
        for key in self.loss_components:
            self.loss_components[key] += batch_loss_components[key]
        return batch_loss, batch_loss_components

    def average_cumulative_loss(self, num_batches):
        self.loss = self.loss / num_batches
        for key in self.loss_components:
            self.loss_components[key] = self.loss_components[key] / num_batches

    def compute_metrics(self):
        metrics = {"loss": self.loss}
        metrics.update(self.loss_components)
        return metrics
