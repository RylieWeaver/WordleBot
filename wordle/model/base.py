# General
import warnings

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Wordle

    

##############################################
# BASE WORDLE MODEL
##############################################
class WordleModel(nn.Module):
    def __init__(self, device, use_inductive_biases=True):
        super().__init__()
        self.device = device
        self.use_inductive_biases = use_inductive_biases

    def forward(self, states):
        pass

    def forward_flops(self, num_states: int = 1):
        raise NotImplementedError
    
    def backward_flops(self, num_states: int = 1):
        return 2 * self.forward_flops(num_states=num_states)
    
    def _inductive_biases(self, states):
        """
        Make a mask indicating the valid actions based on 
        rules that are always optimal in the game of Wordle.
        
        valid_mask: [B, *, V] 
        - A 'False' boolean indicates that an action does not
        respect the predefined inductive bias.
        """
        # Setup
        guessed_mask = states["guessed_mask"].bool()                    # [B, *, V]
        target_mask = states["total_target_mask"].bool()                # [B, *, V]
        last_guess = states["last_guess"]
        device = target_mask.device
        valid_mask = torch.ones_like(                                   # [B, *, V]
            target_mask, dtype=torch.bool, device=device
        )

        # 1) Do not do repeat guesses (unless it was correct)
        valid_mask = valid_mask & (~guessed_mask | target_mask)         # [B, *, V]

        # 2) Always guess a possible target word for last guess
        valid_mask = torch.where(
            last_guess.bool().unsqueeze(-1),                            # [B, *, 1]
            valid_mask & target_mask,                                   # [B, *, V] (target only)
            valid_mask                                                  # [B, *, V] (unchanged)
        )

        # 3) If there are two or fewer possible targets, 
        #    choose from those
        mask = (target_mask.float().sum(dim=-1) <= 2).unsqueeze(-1)     # [B, *, 1]
        valid_mask = torch.where(mask, target_mask, valid_mask)         # [B, *, V]

        return valid_mask
    
    def _mask_norm_probs(self, probs, valid_mask=None):
        """
        Apply a mask on probs and normalize.

        Inputs:
        - probs: [B, *, V]
        - valid_mask: [B, *, V] (optional)

        Returns:
        - normalized_probs: [B, *, V]
        """
        # Setup
        device = probs.device
        uniform = torch.ones_like(probs, device=device, dtype=torch.float32) / probs.shape[-1]      # [B, *, V]
        if valid_mask is None:
            valid_mask = torch.ones_like(probs, device=device, dtype=torch.bool)                    # [B, *, V]
        valid_mask = valid_mask.float()

        # Mask
        probs = probs * valid_mask

        # Set the probabilities of the zero rows (should only happen for inactive episodes, 
        # but can sometimes happen from very small values in the softmax probability) option
        # 1 is masked probs, option 2 is valid actions, and option 3 is uniform distribution
        zero_idx = (probs.sum(dim=-1) < 1e-6)  # [batch_size, *]
        if zero_idx.any():
            warnings.warn(f"Found {zero_idx.sum()} rows with all zero probabilities.")
            has_valid = (valid_mask[zero_idx].sum(dim=-1, keepdim=True) > 0.99)                     # [zero_idx, 1]
            uniform = (
                torch.ones_like(probs[zero_idx], device=device, dtype=torch.float32) /              # [zero_idx, *, V]
                probs[zero_idx].sum(dim=-1, keepdim=True).clamp_min(1e-9)
              )
            probs[zero_idx] = torch.where(has_valid, valid_mask[zero_idx], uniform)                 # [zero_idx, *, V]

        # Normalize the probabilities
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)                             # [B, *, V]
        while (probs.sum(-1) - 1.0).abs().max() > 1e-6:
            warnings.warn("Probabilities sum to more or less than 1.0")
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        return probs
    
    def _make_probs(self, logits, alpha, temperature, valid_mask=None):
        """
        Inputs:
        - logits: [B, *, V]
        - alpha: mixing coefficient for uniform distribution
        - temperature: temperature for softmax
        - valid_mask: [B, *, V] (optional)

        Returns:
        - policy_probs: [B, *, V] (just from the model)
        - mixed_probs: [B, *, V] (after uniform mixing)
        """
        # Setup
        device = logits.device
        if valid_mask is None:
            valid_mask = torch.ones_like(logits, device=device, dtype=torch.bool)               # [B, *, V]
        valid_mask = valid_mask.float()

        # Softmax with temperature
        scaled_logits = logits / temperature
        policy_probs = F.softmax(scaled_logits, dim=-1)

        # Uniform distribution over valid actions for alpha-mixing
        uniform_probs = valid_mask / valid_mask.sum(dim=-1, keepdim=True).clamp_min(1e-9)       # [B, *, V]
        mixed_probs = alpha * uniform_probs + (1 - alpha) * policy_probs

        # Mask and normalize
        policy_probs_masked = self._mask_norm_probs(policy_probs, valid_mask)
        mixed_probs_masked = self._mask_norm_probs(mixed_probs, valid_mask)

        # Collate outputs
        probs = {
            "policy_probs": policy_probs, 
            "policy_probs_masked": policy_probs_masked, 
            "mixed_probs": mixed_probs, 
            "mixed_probs_masked": mixed_probs_masked
        }
        return probs

    def _valid_actions(self, states):
        if self.use_inductive_biases:
            return self._inductive_biases(states)
        return torch.ones_like(states["total_target_mask"], dtype=torch.bool)
    
    def predict(self, states, alpha, temperature):
        logits, values = self.forward(states)                                   # logits: [B, *, V] | values: [B, *]
        valid_mask = self._valid_actions(states)
        probs = self._make_probs(logits, alpha, temperature, valid_mask)
        return probs, values
    
    def sample_with_values(self, states, alpha, temperature, argmax=False):
        # Predict
        logits, values = self.forward(states)                                   # logits: [B, *, V] | values: [B, *]
        valid_mask = self._valid_actions(states)
        probs = self._make_probs(logits, alpha, temperature, valid_mask)

        # Unpack probs
        policy_probs = probs["policy_probs"]                    # [B, *, V]
        policy_probs_masked = probs["policy_probs_masked"]      # [B, *, V]
        mixed_probs = probs["mixed_probs"]                      # [B, *, V]
        mixed_probs_masked = probs["mixed_probs_masked"]        # [B, *, V]
        ppo_probs = policy_probs_masked                         # [B, *, V]


        # Select actions
        if not argmax:
            *dims, V = mixed_probs_masked.shape
            guess_idx = torch.multinomial(mixed_probs_masked.reshape(-1, V), 1).squeeze(-1)     # [*dims]
            guess_idx = guess_idx.reshape(*dims)                                                # [B, *]
        else:
            _, guess_idx = torch.topk(policy_probs_masked, k=1, dim=-1)                         # [B, *, 1]
            guess_idx = guess_idx.squeeze(-1)                                                   # [B, *]
        guess_mask = F.one_hot(
            guess_idx, num_classes=probs["mixed_probs_masked"].shape[-1]
        ).bool()                                                                                # [B, *, V]

        actions = {
            "policy_probs": policy_probs,                       # [B, *, V]
            "policy_probs_masked": policy_probs_masked,         # [B, *, V]
            "mixed_probs": mixed_probs,                         # [B, *, V]
            "mixed_probs_masked": mixed_probs_masked,           # [B, *, V]
            "guess_idx": guess_idx,                             # [B, *]
            "guess_mask": guess_mask,                           # [B, *, V]
            "valid_mask": valid_mask                            # [B, *, V]
        }
        return actions, values
    
    def sample(self, states, alpha, temperature, argmax=False):
        actions, _ = self.sample_with_values(states, alpha, temperature, argmax=argmax)
        return actions


class DotWordleModel(WordleModel):
    def _build_guess_k(self):
        raise NotImplementedError
    
    @torch.no_grad()
    def refresh_static_cache(self):
        was_training = self.training
        nn.Module.train(self, False)
        try:
            self.guess_k_static.copy_(self._build_guess_k())
        finally:
            nn.Module.train(self, was_training)
        return self

    @torch.no_grad()
    def eval(self):
        nn.Module.train(self, False)
        self.guess_k_static.copy_(self._build_guess_k())
        return self

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self.refresh_static_cache()
        return result
