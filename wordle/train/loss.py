# General
from typing import Tuple, List

# Torch
import torch
import torch.nn.functional as F

# Wordle
from wordle.utils import Config, to_float



class EnumLossComponent:
    ACTOR = "actor"
    CRITIC = "critic"
    ENTROPY = "entropy"
    KL_REG = "kl_reg"
    KL_GUIDE = "kl_guide"
    KL_BEST = "kl_best"


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
        loss_weights = {
            EnumLossComponent.ACTOR: 1.0,
            EnumLossComponent.CRITIC: 5.0,
            EnumLossComponent.ENTROPY: 0.00,
            EnumLossComponent.KL_REG: 1.0,
            EnumLossComponent.KL_GUIDE: 0.25,
            EnumLossComponent.KL_BEST: 0.10,
        }
        return loss_weights


class WordleLoss:
    def __init__(self, cfg: WordleLossConfig):
        # Read
        self.cfg = cfg
        self.ratio_prob_clip = self.cfg.ratio_prob_clip
        self.min_prob_clip = self.cfg.min_prob_clip
        self.max_prob_clip = self.cfg.max_prob_clip
        self.eps = self.cfg.eps
        self.clamp = self.cfg.clamp
        # Init objects
        self.init_cumulative_loss()
    
    def log_normalize(self, probs):     # [B, *, V]
        probs = probs + self.eps        # Avoid log(0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.clamp)  # Normalize
        return torch.log(probs)

    def clip_grad_range(self, probs):
        # NOTE: It is sometimes beneficialy to clip grads so that the model can focus
        # on learning new things rather than doubling down on one successful motif.
        keep = (((probs >= self.min_prob_clip) & (probs <= self.max_prob_clip)).float())
        return probs * keep + (1 - keep) * probs.detach()
    
    def clip_prob_ratio(self, ratio):
        keep = (((ratio >= 1 - self.ratio_prob_clip) & (ratio <= 1 + self.ratio_prob_clip)).float())
        return ratio * keep + (1 - keep) * ratio.detach()
    
    def group_norm_advantages(self, advantages, active_mask):                   # [B, G, R] (both)
        # Calculate stats only over active games
        masked_advantages = advantages * active_mask                            # [B, G, R]
        sum_adv = masked_advantages.sum(dim=2, keepdim=True)                    # [B, G, 1]
        num_active = active_mask.sum(dim=2, keepdim=True).clamp_min(self.eps)   # [B, G, 1]
        mean_adv = (sum_adv / num_active)                                       # [B, G, 1]
        diff_adv = (advantages - mean_adv) * active_mask
        sum_square_adv = (diff_adv.pow(2)).sum(dim=2, keepdim=True)
        std_adv = (sum_square_adv / num_active).sqrt().clamp_min(self.eps)
        # Only apply normalization when there are enough active games
        # (otherwise, std would be 0 and cause problems)
        zero = torch.zeros_like(mean_adv)
        one = torch.ones_like(std_adv)
        mean_adv = torch.where(num_active <= 1, zero, mean_adv).detach()  # skip mean normalization when not enough active games
        std_adv = torch.where(num_active <= 1, one, std_adv).detach()  # default to 1.0 std when not enough active games
        advantages = (advantages - mean_adv) / std_adv
        advantages_active = advantages[active_mask]
        return advantages_active
    
    def make_loss(self):
        loss = 0.0
        loss_components = {
            EnumLossComponent.ACTOR: 0.0,
            EnumLossComponent.CRITIC: 0.0,
            EnumLossComponent.ENTROPY: 0.0,
            EnumLossComponent.KL_REG: 0.0,
            EnumLossComponent.KL_GUIDE: 0.0,
            EnumLossComponent.KL_BEST: 0.0
        }
        return loss, loss_components
    
    def init_cumulative_loss(self):
        self.loss, self.loss_components = self.make_loss()

    def measure_grad_norms(
            self, model, states, actions, responses, 
            probs, ref_probs, best_probs
        ) -> Tuple[float, float, float, float, float, float]:
        """
        Run backward passes to measure the gradient-norm of:
            1) actor_coef * actor_loss
            2) critic_coef * critic_loss
            3) entropy_coef * entropy_loss
            4) kl_reg_coef * kl_reg_loss
            5) kl_guide_coef * kl_guide_loss
            6) kl_best_coef * kl_best_loss
        """

        # Define helper
        def _mean_grad_norm(scaled_loss: torch.Tensor) -> float:
            grads: List[torch.Tensor] = torch.autograd.grad(
                scaled_loss,
                [p for p in model.parameters() if p.requires_grad],
                retain_graph=True,       # keep the graph alive for the upcoming backward
                create_graph=False,
                allow_unused=True,
            )
            grads = [g for g in grads if g is not None]
            if not grads:                     # safeguard (shouldn’t happen)
                return 0.0
            return torch.stack([g.norm() for g in grads]).mean().item()
        # -----------------------------------------------------------------

        # Calculate
        actor_norm = critic_norm = entropy_norm = kl_reg_norm = kl_guide_norm = kl_best_norm = 0.0
        actor_norm = _mean_grad_norm(
            self.cfg.loss_weights[EnumLossComponent.ACTOR] * 
            self.calculate_loss_components(states, actions, responses, probs, ref_probs, best_probs)[0]
        ) if self.cfg.loss_weights[EnumLossComponent.ACTOR] != 0.0 else 0.0
        critic_norm = _mean_grad_norm(
            self.cfg.loss_weights[EnumLossComponent.CRITIC] * 
            self.calculate_loss_components(states, actions, responses, probs, ref_probs, best_probs)[1]
        ) if self.cfg.loss_weights[EnumLossComponent.CRITIC] != 0.0 else 0.0
        entropy_norm = _mean_grad_norm(
            self.cfg.loss_weights[EnumLossComponent.ENTROPY] * 
            self.calculate_loss_components(states, actions, responses, probs, ref_probs, best_probs)[2]
        ) if self.cfg.loss_weights[EnumLossComponent.ENTROPY] != 0.0 else 0.0
        kl_reg_norm = _mean_grad_norm(
            self.cfg.loss_weights[EnumLossComponent.KL_REG] * 
            self.calculate_loss_components(states, actions, responses, probs, ref_probs, best_probs)[3]
        ) if self.cfg.loss_weights[EnumLossComponent.KL_REG] != 0.0 else 0.0
        kl_guide_norm = _mean_grad_norm(
            self.cfg.loss_weights[EnumLossComponent.KL_GUIDE] * 
            self.calculate_loss_components(states, actions, responses, probs, ref_probs, best_probs)[4]
        ) if self.cfg.loss_weights[EnumLossComponent.KL_GUIDE] != 0.0 else 0.0
        kl_best_norm = _mean_grad_norm(
            self.cfg.loss_weights[EnumLossComponent.KL_BEST] * 
            self.calculate_loss_components(states, actions, responses, probs, ref_probs, best_probs)[5]
        ) if self.cfg.loss_weights[EnumLossComponent.KL_BEST] != 0.0 else 0.0

        # Show
        print(
            f"[Grad Norms] Actor: {actor_norm:.6f} | Critic: {critic_norm:.6f} | "
            f"Entropy: {entropy_norm:.6f} | KL Reg: {kl_reg_norm:.6f} | "
            f"KL Guide: {kl_guide_norm:.6f} | KL Best: {kl_best_norm:.6f}"
        )

    def calculate_loss_components(self, states, actions, responses, probs, ref_probs, best_probs) -> Tuple[torch.Tensor, dict]:
        # Unpack
        active_mask = states["active_mask"][:, :-1, ...]                # [B, G+1, *] --> [B, G, *]
        ref_probs = ref_probs["policy_probs"][:, :-1, ...]              # [B, G, *, V]
        valid_mask = actions["valid_mask"]                              # [B, G, V]
        guess_mask = actions["guess_mask"]                              # [B, G, V]
        policy_probs = probs["policy_probs"]                            # [B, G, *, V]
        masked_probs = probs["policy_probs_masked"]                     # [B, G, *, V]
        best_probs = best_probs["policy_probs"][:, :-1, ...]            # [B, G, *, V]
        advantages = responses["advantages"]                            # [B, G, *]

        # Freeze gradient for probs outside threshold
        # (can prevent kl-guide from getting too much of the gradient to go all the way to 0 or 1)
        range_clipped_probs = self.clip_grad_range(policy_probs)

        # Mask for only active turns
        ref_probs_active = ref_probs[active_mask]
        probs_active = policy_probs[active_mask]
        range_clipped_probs_active = range_clipped_probs[active_mask]
        masked_probs_active = masked_probs[active_mask]
        best_probs_active = best_probs[active_mask]
        advantages_active = advantages[active_mask]

        # Prob distribution log terms
        ref_log_probs_active = self.log_normalize(ref_probs_active)
        log_probs_active = self.log_normalize(probs_active)
        clipped_log_probs_active = self.log_normalize(range_clipped_probs_active)
        masked_log_probs_active = self.log_normalize(masked_probs_active)
        best_log_probs_active = self.log_normalize(best_probs_active)

        # # Entropy regularization
        # entropy_probs = policy_probs * valid_mask  # should not include the probabilities which we deem invalid
        # entropy_probs = entropy_probs / entropy_probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        # entropy_log_probs = self.log_normalize(entropy_probs)
        # entropies = -torch.sum(entropy_probs * entropy_log_probs, dim=-1)
        # entropies_active = entropies[active_mask]

        # Prob ratio of chosen actions for the actor loss
        active_guess_mask = guess_mask[active_mask]
        chosen_ref_log_probs = ref_log_probs_active[active_guess_mask]
        chosen_log_probs = log_probs_active[active_guess_mask]
        prob_ratio = torch.exp(chosen_log_probs - chosen_ref_log_probs)
        clipped = torch.clamp(prob_ratio, 1 - self.ratio_prob_clip, 1 + self.ratio_prob_clip)
        # prob_ratio = self.clip_prob_ratio(prob_ratio)

        # Get normalized advantages by group
        # NOTE: Detach advantage grad for actor loss because that would 
        #       train the critic to change the advantages for actor loss!
        detached_advantages = advantages.clone().detach()
        policy_advantages_active = self.group_norm_advantages(detached_advantages, active_mask)

        # Critic loss is computed with the advantages before normalization
        critic_losses = advantages_active.pow(2)

        surr1 = prob_ratio * policy_advantages_active
        surr2 = clipped * policy_advantages_active
        actor_loss = -torch.min(surr1, surr2).mean()

        # Actor loss
        # actor_loss = -(prob_ratio * policy_advantages_active).mean()
        # Critic loss
        critic_loss = critic_losses.mean()
        # Entropy loss
        # entropy_loss = -entropies_active.mean()
        entropy_loss = 0.0
        # KL-Div losses
        ## NOTE: Of the form F.kl_div(input, target) = KL(target || input) = mean(sum(target * (log(target) - log(input)), dim=-1))
        kl_reg_loss = F.kl_div(
            ref_log_probs_active, log_probs_active, reduction='batchmean', log_target=True
        )
        kl_guide_loss = F.kl_div(
            masked_log_probs_active.detach(), clipped_log_probs_active, reduction='batchmean', log_target=True
        )
        kl_best_loss = F.kl_div(
            best_log_probs_active, log_probs_active, reduction='batchmean', log_target=True
        )
        return (actor_loss, critic_loss, entropy_loss, kl_reg_loss, kl_guide_loss, kl_best_loss)

    def inc_loss(self, states, actions, responses, probs, ref_probs, best_probs):
        """
        This function starts with a lot of setup to create the various objects 
        needed for calculating the loss components.
        """
        # Setup
        batch_loss, batch_loss_components = self.make_loss()
        losses = self.calculate_loss_components(states, actions, responses, probs, ref_probs, best_probs)

        # Calculate loss components
        batch_loss_components[EnumLossComponent.ACTOR] = to_float(losses[0])
        batch_loss_components[EnumLossComponent.CRITIC] = to_float(losses[1])
        batch_loss_components[EnumLossComponent.ENTROPY] = to_float(losses[2])
        batch_loss_components[EnumLossComponent.KL_REG] = to_float(losses[3])
        batch_loss_components[EnumLossComponent.KL_GUIDE] = to_float(losses[4])
        batch_loss_components[EnumLossComponent.KL_BEST] = to_float(losses[5])
        # Total loss
        batch_loss = (
            self.cfg.loss_weights[EnumLossComponent.ACTOR] * losses[0] +
            self.cfg.loss_weights[EnumLossComponent.CRITIC] * losses[1] +
            self.cfg.loss_weights[EnumLossComponent.ENTROPY] * losses[2] +
            self.cfg.loss_weights[EnumLossComponent.KL_REG] * losses[3] +
            self.cfg.loss_weights[EnumLossComponent.KL_GUIDE] * losses[4] +
            self.cfg.loss_weights[EnumLossComponent.KL_BEST] * losses[5]
        )

        # Inc metrics (with coefficients for total_loss)
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
