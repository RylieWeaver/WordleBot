# General
import math
import sys
from tqdm import tqdm
from typing import Union
import warnings


# Torch
import torch
import torch.nn.functional as F

# Wordle
from wordle.data import WordleLoaderConfig, WordleLoader, move_to, tensor_to_words, words_to_tensor
from wordle.utils import Config, expand_var



class SimulatorConfig(Config):
    def __init__(
            self,
            loader_cfg: Union[WordleLoaderConfig, dict] = None,
            correct_reward: float = 0.1,
            correct_blend_factor: float = 1.0,
            max_guesses: int = 6,
            gamma: float = 0.20,
            lam: float = 0.95,
            m: int = 3,
            reward_blend_factor: float = 1.0,
            value_blend_factor: float = 1.0,
            advantage_type: str = "reward-telescoped-value-baseline",
            adv_mean_reduce_dims: Union[tuple, list, int, None] = (2,),
            adv_std_reduce_dims: Union[tuple, list, int, None] = (0, 2),
            fp_dtype: str = 'float32',
        ):
        self.loader_cfg = loader_cfg if loader_cfg is not None else WordleLoaderConfig()
        if isinstance(self.loader_cfg, dict):
            self.loader_cfg = WordleLoaderConfig(**self.loader_cfg)
        self.correct_reward = correct_reward
        self.correct_blend_factor = correct_blend_factor
        self.max_guesses = max_guesses
        self.gamma = gamma
        self.lam = lam
        self.m = m
        self.reward_blend_factor = reward_blend_factor
        self.value_blend_factor = value_blend_factor
        self.advantage_type = self._as_advantage_type(advantage_type)
        self.adv_mean_reduce_dims = self._as_reduce_dims(adv_mean_reduce_dims)
        self.adv_std_reduce_dims = self._as_reduce_dims(adv_std_reduce_dims)
        self.fp_dtype = fp_dtype

    def _as_advantage_type(self, advantage_type):
        aliases = {
            "reward-telecoped": "reward-telescoped",
            "reward-telecoped-value-baseline": "reward-telescoped-value-baseline",
        }
        advantage_type = aliases.get(advantage_type, advantage_type)
        valid = {
            "gae",
            "reward-total",
            "reward-telescoped",
            "reward-telescoped-value-baseline",
        }
        if advantage_type not in valid:
            raise ValueError(f"Unknown advantage_type: {advantage_type}. Valid options are {sorted(valid)}")
        return advantage_type

    def _as_reduce_dims(self, reduce_dims):
        if reduce_dims is None:
            return None
        if isinstance(reduce_dims, int):
            return (reduce_dims,)
        return tuple(reduce_dims)


class Simulator:
    def __init__(self, cfg: SimulatorConfig):
        """
        The '*' dimension in tensor shape indicates any number of the following dimensions:
        - G: t between 0 and max_guesses
        - R: target word repetition (aka group size for GRPO)
        - m: number of candidate target words sampled for a given state
        """
        # Read
        self.cfg = cfg
        self.correct_reward = self.cfg.correct_reward
        self.correct_blend_factor = self.cfg.correct_blend_factor
        self.max_guesses = self.cfg.max_guesses
        self.gamma = self.cfg.gamma
        self.lam = self.cfg.lam
        self.m = self.cfg.m
        self.reward_blend_factor = self.cfg.reward_blend_factor
        self.value_blend_factor = self.cfg.value_blend_factor
        self.advantage_type = self.cfg.advantage_type
        self.adv_mean_reduce_dims = self.cfg.adv_mean_reduce_dims
        self.adv_std_reduce_dims = self.cfg.adv_std_reduce_dims
        self.fp_dtype = getattr(torch, self.cfg.fp_dtype)
        self.eps = 1e-8

        # Instantiate
        self.set_loader(self.cfg.loader_cfg)

        # Aliases
        self.G = self.max_guesses
        self.R_c = self.correct_reward
        self.c = self.correct_blend_factor

    def set_loader(self, loader_cfg: WordleLoaderConfig):
        self.cfg.loader_cfg = loader_cfg
        self.loader = WordleLoader(self.cfg.loader_cfg)
        self.target_vocab_size = len(self.cfg.loader_cfg.target_vocab)
        self.total_vocab_size = len(self.cfg.loader_cfg.target_vocab) + len(self.cfg.loader_cfg.nontarget_vocab)

    def _append_dict(self, dict1, dict2):
        for k in dict2.keys():
            if k not in dict1:
                dict1[k] = []
            dict1[k].append(dict2[k])
    
    def _stack_dict(self, dict, dim):
        for k in dict.keys():
            dict[k] = torch.stack(dict[k], dim=dim)
        return dict
    
    def _cat_dict(self, dict, dim):
        for k in dict.keys():
            dict[k] = torch.cat(dict[k], dim=dim)
        return dict

    def _init_states(self, batch_size, device="cpu"):
        with torch.no_grad():
            # Setup
            B = batch_size  # Could be different for last batch in the epoch
            R = self.loader.repeats
            V = self.loader.V
            T = self.loader.T
            fp_dtype = self.fp_dtype
            
            # Initialize
            entropy = math.log2(max(T, 1))
            # NOTE: Even though there is some redundancy in "t" and "last_guess", keeping 
            # the batch shape is helpful to collect minibatches and later index the dataset
            states = {
                "t": torch.ones((B, R), dtype=torch.int64, device=device),                      # [B, R]
                "last_guess": torch.zeros((B, R), dtype=torch.bool, device=device),             # [B, R]
                "alphabet": torch.zeros((B, R, 26, 11), dtype=fp_dtype, device=device),         # [B, R, 26, 11]
                "active_mask": torch.ones((B, R), dtype=torch.bool, device=device),             # [B, R]
                "guessed_mask": torch.zeros((B, R, V), dtype=torch.bool, device=device),        # [B, R, V]
                "target_mask": torch.ones((B, R, T), dtype=torch.bool, device=device),          # [B, R, T]
                "entropy": torch.full((B, R), entropy, dtype=fp_dtype, device=device)           # [B, R]
            }
            states["total_target_mask"] = F.pad(states["target_mask"], (0, V - T))              # [B, R, V]
            return states


    def _update_alphabet(self, data, states, actions):
        """
        Update the states with tensor operations.

        Inputs:
        - alphabet: [B, *, 26, 11]
        - guess: [B, *, 5]
        - target_idx: [B, *, 5]

        Created:
        - alphabet_tensor: [26, 1]
        - guess_loc: [B, *, 26, 5] mask for guessed letter locations
        - target_loc: [B, *, 26, 5] mask for target letter locations
        - guessed: [B, *, 26, 1] number of times a letter appears in our guessed word
        - exist: [B, *, 26, 1] number of times a letter appears in the target word
        - letter_count: [B, *, 26, 1] minimum of guessed and exist. We know a letter must appear this many times in the target!
        - green_mask: [B, *, 26, 5] boolean mask for correct letter guesses AND placement
        - not_exist: [B, *, 26, 5] boolean mask for guessing a letter that is not in the target anywhere
        - found_not: [B, *, 26, 5] boolean mask for guessing a letter that is in the target but not in the location
        - grey_mask: [B, *, 26, 5] boolean mask for incorrect letter guesses

        Returns:
        - alphabet: [B, *, 26, 11] updated
        - 
        """
        # Setup
        target_idx = data["target"]["batch_idx"]                                    # [B, *] (the idx of the true target word in the target vocab)
        guess_idx = actions["guess_idx"]                                            # [B, *] (the idx of the guessed word in the total vocab)
        target_tensor = data["target"]["tensor"][target_idx]                        # [B, *, 5]
        guess_tensor = data["total"]["tensor"][guess_idx]                           # [B, *, 5]
        alphabet = states["alphabet"]                                               # [B, *, 26, 11]

        # OHE guess/target information
        guess_loc = F.one_hot(
            guess_tensor, num_classes=26
        ).transpose(-2, -1).bool()                                                  # [B, *, 5, 26] -> [B, *, 26, 5]
        target_loc = F.one_hot(
            target_tensor, num_classes=26
        ).transpose(-2, -1).bool()                                                  # [B, *, 5, 26] -> [B, *, 26, 5]
    
        # Counting occurrences
        guessed = guess_loc.float().sum(dim=-1, keepdim=True)                       # [B, *, 26, 1]
        exist = target_loc.float().sum(dim=-1, keepdim=True)                        # [B, *, 26, 1]
        letter_count = torch.min(guessed, exist)                                    # [B, *, 26, 1]

        # Compute first order information (directly from guess/target)
        ## Matches
        green_mask = guess_loc & target_loc                                         # [B, *, 26, 5]
        ## Non-matches
        not_exist = ((guessed > 0) & (exist == 0)).expand_as(guess_loc)             # [B, *, 26, 5]
        found_not = guess_loc & ~target_loc                                         # [B, *, 26, 5]
        grey_mask = not_exist | found_not                                           # [B, *, 26, 5]
        ## Combine
        letter_count = torch.max(alphabet[..., :1], letter_count)                   # [B, *, 26, 1]
        green_mask = green_mask | alphabet[..., 1:6].bool()                         # [B, *, 26, 5]
        grey_mask = grey_mask | alphabet[..., 6:].bool()                            # [B, *, 26, 5]

        # Compute second order information (combining known info)
        ## Green implies all others grey
        occupied = green_mask.any(dim=-2, keepdim=True) & ~green_mask               # [B, *, 26, 5]
        grey_mask = grey_mask | occupied                                            # [B, *, 26, 5]
        ## All others grey implies green
        no_others = (grey_mask.float().sum(dim=-2, keepdim=True) == 25)             # [B, *, 1, 5]
        implied_greens = no_others & ~grey_mask                                     # [B, *, 26, 5]
        green_mask = green_mask | implied_greens                                    # [B, *, 26, 5]

        # Combine and update
        alphabet = torch.cat(                                                       # [B, *, 26, 11]
            [letter_count, green_mask, grey_mask], dim=-1
        ).float()
        return alphabet
    
    def _calculate_entropy(self, data, states):
        """
        Inputs:
        - target: [B, *, T, 26, 11] (indicates total OHE information given each target word)
          Being less than or equal to the alphabet states indicates consistency with that target word!
        
        It is assumed that entropy is calculated as uniform distribution over words.
        """
        # Setup
        dims = states["alphabet"].shape[:-2]                                        # [B, *]
        target = data["target"]["states"]                                           # [T, 26, 11]
        target = target.view(*([1] * (len(dims))), *target.shape)                   # [1, ..., T, 26, 11]
        alphabet = states["alphabet"].unsqueeze(-3)                                 # [B, *, 1, 26, 11]

        # Get entropy
        target_mask = (alphabet <= target).all(dim=[-2, -1])                        # [B, *, T, 26, 11] -> [B, *, T]
        entropies = torch.log2(target_mask.sum(dim=-1).float().clamp_min(1.0))      # [B, *]  (clamp to avoid log2(0))
        return target_mask, entropies

    def _calculate_rewards(self, data, states, actions):
        """
        H stands for entropy and is assumed to be over a uniform distribution.
        """
        # Setup
        idx = data["total"]["batch_idx"]                                                    # [B, *] (the idx of the true target word in the total vocab)
        total_target_mask = states["total_target_mask"]                                     # [B, *, T]
        target_mask = states["target_mask"]                                                 # [B, *, T]
        H_old = states["entropy"]                                                           # [B, *]
        H_max = math.log2(data["target"]["size"])                                           # [B, *]
        G = self.max_guesses
        guess_idx = actions["guess_idx"]                                                    # [B, *]

        # Calculate
        ## Entropy
        new_target_mask, H_new = self._calculate_entropy(data, states)                      # [B, *, T] | [B, *]
        entropy_reward = (H_old - H_new)/H_max                                              # Scales episodic rewards to [0, 1] invariant of vocab size
        ## Guess
        guess_cost = 1 / G
        ## Correct
        correct_cand = total_target_mask.gather(-1, guess_idx.unsqueeze(-1)).squeeze(-1)    # [B, *]
        count_cand = target_mask.sum(dim=-1)                                                # [B, *]
        correct = (idx == guess_idx).float()                                                # [B, *]
        correct_reward = (                                                                  # [B, *]
            self.R_c * (((1-self.c) * correct) + (self.c/count_cand * correct_cand))
        )
        rewards = entropy_reward - guess_cost + correct_reward                              # [B, *]
        return rewards, H_new, new_target_mask, correct
    
    def _sample_targets(self, target_mask):
        """
        Sample 'm' candidates from the target mask.

        Variables:
        - m: number of possible target words to sample for each action
        - target_mask: [B, *, T]
        - idx: [B, *, m]
        """
        # Setup
        base_shape = target_mask.shape[:-1]        # [B, *]
        m = self.m
        device = target_mask.device

        # Make a mask for when there exist 'm' candidates
        mask_f = target_mask.float()
        enough = mask_f.sum(dim=-1) >= m

        # Sample w/o replacement when we have enough candidates, else with replacement
        idx = torch.empty((*base_shape, m), dtype=torch.long, device=device)
        idx[enough] = torch.multinomial(mask_f[enough], m, replacement=False)
        idx[~enough] = torch.multinomial(mask_f[~enough], m, replacement=True)
        return idx
    
    def _target_expansion(self, states, actions, m):
        """
        Expand the states to have an additional dimension of size m for candidate target words.
        """
        idx = self._sample_targets(states["target_mask"])                                           # [B, m]
        cand_data = self.loader._idx2data(idx)                                                      # dict with [B, m] batch idx for target and total
        cand_states = {
            "t": expand_var(states["t"], dim=-1, size=m),                                           # [B, *, m]
            "alphabet": expand_var(states["alphabet"], dim=-3, size=m),                             # [B, *, m, 26, 11]
            "active_mask": expand_var(states["active_mask"], dim=-1, size=m),                       # [B, *, m]
            "guessed_mask": expand_var(states["guessed_mask"], dim=-2, size=m),                     # [B, *, m, V]
            "target_mask": expand_var(states["target_mask"], dim=-2, size=m),                       # [B, *, m, T]
            "total_target_mask": expand_var(states["total_target_mask"], dim=-2, size=m),           # [B, *, m, V]
            "entropy": expand_var(states["entropy"], dim=-1, size=m),                               # [B, *, m]
            "idx": idx,                                                                             # [B, *, m]
        }
        cand_actions = {
            "policy_probs": expand_var(actions["policy_probs"], dim=-2, size=m),                    # [B, *, m, V]
            "policy_probs_masked": expand_var(actions["policy_probs_masked"], dim=-2, size=m),      # [B, *, m, V]
            "mixed_probs": expand_var(actions["mixed_probs"], dim=-2, size=m),                      # [B, *, m, V]
            "mixed_probs_masked": expand_var(actions["mixed_probs_masked"], dim=-2, size=m),        # [B, *, m, V]
            "guess_idx": expand_var(actions["guess_idx"], dim=-1, size=m),                          # [B, *, m]
            "guess_mask": expand_var(actions["guess_mask"], dim=-2, size=m),                        # [B, *, m, V]
            "valid_mask": expand_var(actions["valid_mask"], dim=-2, size=m),                        # [B, *, m, V]
        }
        return cand_data, cand_states, cand_actions

    def step(self, model, data, states, actions):
        """
        Simulate one Wordle step

        Data:
        - target:
            - states: [T, 26, 11]
            - tensor: [T, 5]
        - total:
            - states: [V, 26, 11]
            - tensor: [V, 5]
        
        States:
        - t: [B, *] (int)
        - last_guess: [B, *] (bool)
        - alphabet: [B, *, 26, 11]
        - active_mask: [B, *]
        - guessed_mask: [B, *, V]
        - target_mask: [B, *, T]
        - total_target_mask: [B, *, V]
        - entropy: [B, *]

        Returns:
        - updated states
        - responses:
            - values: [B, *]
            - rewards: [B, *]
            - expected_values: [B, *]
            - expected_rewards: [B, *]
            - correct: [B, *]
        """
        # Setup
        T = data["target"]["size"]
        V = data["total"]["size"]

        # Simulate possible target words given the target mask
        cand_data, cand_states, cand_actions = self._target_expansion(states, actions, self.m)
        cand_data = move_to(cand_data, data["total"]["tensor"].device)
        cand_states = move_to(cand_states, states["alphabet"].device)

        # Update what we can in states before simulating response 
        # (t, alphabet, guessed_mask, active_mask)
        states["t"] = states["t"] + 1
        states["alphabet"] = self._update_alphabet(data, states, actions)           # [B, *, 26, 11]
        states["last_guess"] = (states["t"] == self.max_guesses)                    # [B, *]
        states["guessed_mask"] = states["guessed_mask"] | F.one_hot(                # [B, *, V]
            actions["guess_idx"], num_classes=V
        ).bool()
        cand_states["t"] = cand_states["t"] + 1
        cand_states["alphabet"] = self._update_alphabet(                            # [B, *, m, 26, 11]
            cand_data, cand_states, cand_actions
        )
        states["last_guess"] = (states["t"] == self.max_guesses)                    # [B, *]
        cand_states["guessed_mask"] = cand_states["guessed_mask"] | F.one_hot(      # [B, *, m, V]
            cand_actions["guess_idx"], num_classes=V
        ).bool()

        # Calculate responses
        # NOTE: The model only requires updated t/alphabet for value estimates
        with torch.no_grad():
            _, values = model(states)                                               # [B, *]
            _, cand_values = model(cand_states)                                     # [B, *, m]
        rewards, new_entropy, new_target_mask, correct = (                          # [B, *], [B, *], [B, *, T]
            self._calculate_rewards(data, states, actions)
        )
        cand_rewards, _, _, _ = (                                                   # [B, *, m]
            self._calculate_rewards(cand_data, cand_states, cand_actions)
        )
        # NOTE: Right-padding requires the convention of targets first in the total vocab
        new_total_target_mask = F.pad(new_target_mask, (0, V - T)).bool()           # [B, *, V]

        # Update rest of states
        ## Inactive if was last guess
        active = torch.where(
            (states["t"] >  self.max_guesses),                                      # [B, *]
            torch.zeros_like(states["active_mask"], dtype=torch.bool),              # [B, *]
            states["active_mask"]                                                   # [B, *]
        )
        ## Inactive if was correct (a.k.a. won)
        active = active & (~correct.bool())                                         # [B, *]
        states["active_mask"] = active                                              # [B, *]
        states["target_mask"] = new_target_mask                                     # [B, *, T]
        states["total_target_mask"] = new_total_target_mask                         # [B, *, V]
        states["entropy"] = new_entropy                                             # [B, *]

        # Collect responses
        responses = {
            "values": values,                                                       # [B, *]
            "rewards": rewards,                                                     # [B, *]
            "expected_values": cand_values.mean(dim=-1),                            # [B, *]
            "expected_rewards": cand_rewards.mean(dim=-1),                          # [B, *]
            "correct": correct                                                      # [B, *]
        }

        return states, responses
    
    def _masked_mean(self, x, mask, reduce_dims):
        if reduce_dims is None:
            return torch.zeros((), dtype=x.dtype, device=x.device), None
        count = mask.sum(dim=reduce_dims, keepdim=True)
        mean = (x * mask).sum(dim=reduce_dims, keepdim=True) / count.clamp_min(1.0)
        mean = torch.where(count > 0, mean, torch.zeros_like(mean))
        return mean, count
    
    def _masked_std(self, x, mask, reduce_dims):
        if reduce_dims is None:
            return torch.ones((), dtype=x.dtype, device=x.device), None
        mean, count = self._masked_mean(x, mask, reduce_dims)
        ss_res = (((x - mean) * mask) ** 2).sum(dim=reduce_dims, keepdim=True)
        std = torch.sqrt(ss_res / count.clamp_min(1.0)).clamp_min(self.eps)
        std = torch.where(count > 0, std, torch.ones_like(std))
        return std, count
    
    def norm_advantages(self, advantages, active_mask):                              # [B, G, R] (both)
        adv_mean, mean_count = self._masked_mean(                                    # [*, *, *]
            advantages, active_mask, self.adv_mean_reduce_dims
        )
        adv_std, std_count = self._masked_std(                                       # [*, *, *]
            advantages, active_mask, self.adv_std_reduce_dims
        )

        # Default normalization to 0/1 when there aren't enough active games
        # NOTE: Skip zero active because then there's just no episode
        min_adv_count = 2
        if mean_count is not None:
            mean_leq_min = (mean_count > 0) & (mean_count < min_adv_count)
            if mean_leq_min.any():
                warnings.warn(
                    f"Mean advantage count is < {min_adv_count} for \
                    {mean_leq_min.sum().item()} groups. Defaulting to 0 mean advantage."
                )
                adv_mean = torch.where(mean_leq_min, torch.zeros_like(adv_mean), adv_mean).detach()
        if std_count is not None:
            std_leq_min = (std_count > 0) & (std_count < min_adv_count)
            if std_leq_min.any():
                warnings.warn(
                    f"Std advantage count is < {min_adv_count} for \
                    {std_leq_min.sum().item()} groups. Defaulting to 1 std advantage."
                )
                adv_std = torch.where(std_leq_min, torch.ones_like(adv_std), adv_std).detach()

        # Apply normalization
        norm_advantages = ((advantages - adv_mean) / adv_std) * active_mask           # [B, G, R]
        return norm_advantages, adv_mean, adv_std
    
    def calculate_advantages(self, responses, active_mask):
        # Read
        rewards = responses["rewards"]                      # [B, G, *]
        values = responses["values"]                        # [B, G+1, *]
        expected_rewards = responses["expected_rewards"]    # [B, G, *]
        expected_values = responses["expected_values"]      # [B, G, *]
        active = active_mask[:, :-1, ...].float()           # [B, G, *]
        next_active = active_mask[:, 1:, ...].float()       # [B, G, *]
        
        # Setup
        B, _, *dims = active_mask.shape
        G = self.max_guesses
        device = rewards.device

        # Blend and mask the tensors
        # NOTE: Important for advantages to not be influenced by values/rewards after an episode finishes!
        blended_rewards = (                                                             # [B, G, *]
            self.reward_blend_factor * expected_rewards +
            (1 - self.reward_blend_factor) * rewards
        ) * active
        blended_next_values = (                                                         # [B, G, *]
            self.value_blend_factor * expected_values +
            (1 - self.value_blend_factor) * values[:, 1:, ...]
        ).detach() * next_active
        current_values = (values[:, :-1, ...] * active).detach()                        # [B, G, *]

        # Compute returns and advantages
        if self.advantage_type == "gae":
            advantages = torch.zeros((B, G, *dims), dtype=self.fp_dtype, device=device)
            gae = torch.zeros((B, *dims), dtype=self.fp_dtype, device=device)
            for t in reversed(range(G)):
                delta = (
                    blended_rewards[:, t, ...] +
                    self.gamma * blended_next_values[:, t, ...] -
                    current_values[:, t, ...]
                )
                gae = delta + self.gamma * self.lam * next_active[:, t, ...] * gae
                advantages[:, t, ...] = gae
            advantages = advantages * active
            returns = (advantages + current_values) * active
        elif self.advantage_type == "reward-total":
            returns = blended_rewards.sum(dim=1, keepdim=True).expand(B, G, *dims) * active
            advantages = returns
        elif self.advantage_type in {"reward-telescoped", "reward-telescoped-value-baseline"}:
            returns = torch.zeros((B, G, *dims), dtype=self.fp_dtype, device=device)
            running_return = torch.zeros((B, *dims), dtype=self.fp_dtype, device=device)
            for t in reversed(range(G)):
                running_return = blended_rewards[:, t, ...] + self.gamma * running_return
                returns[:, t, ...] = running_return
            returns = returns * active
            if self.advantage_type == "reward-telescoped-value-baseline":
                advantages = (returns - current_values) * active
            else:
                advantages = returns
        else:
            raise ValueError(f"Unknown advantage_type: {self.advantage_type}")

        norm_advantages, adv_mean, adv_std = self.norm_advantages(advantages, active)
        responses["returns"] = returns.detach()                                             # [B, G, *]
        responses["advantages"] = advantages.detach()                                       # [B, G, *]
        responses["norm_advantages"] = norm_advantages                                      # [B, G, *]
        return responses
    
    def collect_episodes_mb(self, model, data, alpha, temperature, argmax=False):
        """We keep the entire episodes on CPU, then states, actions, responses on Model device"""
        with torch.no_grad():
            # Setup
            G = self.max_guesses
            B = data["target"]["batch_idx"].shape[0]  # Could be different for last batch in the epoch
            
            # Initialize history (states/values)
            episodes = move_to({"states": {}, "actions": {}, "responses": {}}, "cpu")
            states = self._init_states(batch_size=B, device=model.device)
            states["idx"] = data["target"]["batch_idx"].to(model.device)    # [B, *] (the idx of the true target word in the target vocab)
            _, values = model(states)
            self._append_dict(episodes["states"], move_to(states, "cpu"))
            self._append_dict(episodes["responses"], move_to({"values": values}, "cpu"))

            # Device transfer for data
            data = move_to(data, model.device)

            # Roll out to max guesses
            # NOTE: sadly we can't early-stop the loop because we need consistent shapes to stack/concatenate
            #       the episodes after collecting.
            # while (states["t"] <= G).all() and states["active_mask"].any():
            for t in range(1, G+1):
                # Select actions = f(states)
                actions = model.sample(states, alpha, temperature, argmax=argmax)

                # Simulate responses = f(states, actions)
                states, responses = self.step(model, data, states, actions)

                # Store
                self._append_dict(episodes["states"], move_to(states, "cpu"))
                self._append_dict(episodes["actions"], move_to(actions, "cpu"))
                self._append_dict(episodes["responses"], move_to(responses, "cpu"))
            # Stack
            episodes["states"] = self._stack_dict(episodes["states"], dim=1)
            episodes["actions"] = self._stack_dict(episodes["actions"], dim=1)
            episodes["responses"] = self._stack_dict(episodes["responses"], dim=1)
        
        return episodes
    
    def collect_episodes_epoch(self, model, alpha, temperature, argmax=False, desc="Collecting Episodes"):
        # Initialize
        episodes = move_to({"states": {}, "actions": {}, "responses": {}}, "cpu")
        episodes = move_to(episodes, model.device)

        # Collect by minibatch
        for data in tqdm(self.loader, desc=desc, leave=False, disable=not sys.stderr.isatty()):
            episodes_mb = self.collect_episodes_mb(
                model, data, alpha, temperature, argmax=argmax
            )
            self._append_dict(episodes["states"], episodes_mb["states"])
            self._append_dict(episodes["actions"], episodes_mb["actions"])
            self._append_dict(episodes["responses"], episodes_mb["responses"])
        # Stack
        episodes["states"] = self._cat_dict(episodes["states"], dim=0)
        episodes["actions"] = self._cat_dict(episodes["actions"], dim=0)
        episodes["responses"] = self._cat_dict(episodes["responses"], dim=0)    

        # Calculate advantages
        episodes["responses"] = self.calculate_advantages(
            episodes["responses"], episodes["states"]["active_mask"]
        )    
        return episodes

    def process_episodes(
        self,
        model,
        episodes,
        alpha,
        temperature,
    ):
        # Device management
        device = model.device
        episodes = move_to(episodes, device)

        # Unpack episodes
        # NOTE: Unpack most rewards/values from responses. 
        #       Only need values to propagate gradients through the model.
        states = episodes["states"]
        active_mask = states["active_mask"]                 # [B, G+1, *]
        actions = episodes["actions"]
        valid_mask = actions["valid_mask"]                  # [B, G, *, V]
        responses = episodes["responses"]

        # Sample (we must use these values to propagate gradients through the model)
        logits, pred_values = model(states)                                     # [B, G+1, *] | [B, G+1, *]
        logits = logits[:, :-1, ...]                                            # [B, G, *]
        probs = model._make_probs(logits, alpha, temperature, valid_mask)       # dict of [B, G, *, V]

        # Collect new responses
        responses["pred_values"] = (pred_values * active_mask.float())[:, :-1, ...]     # [B, G+1, *]                                               # [B, G, *]
        return probs, responses
