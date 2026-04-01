# General
import math
from tqdm import tqdm
from typing import Union


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
            correct_cand_coeff: float = 1.0,
            max_guesses: int = 6,
            gamma: float = 0.20,
            lam: float = 0.95,
            m: int = 3,
            reward_blend_factor: float = 1.0,
            value_blend_factor: float = 1.0,
            fp_dtype: str = 'float32',
        ):
        self.loader_cfg = loader_cfg if loader_cfg is not None else WordleLoaderConfig()
        if isinstance(self.loader_cfg, dict):
            self.loader_cfg = WordleLoaderConfig(**self.loader_cfg)
        self.correct_reward = correct_reward
        self.correct_cand_coeff = correct_cand_coeff
        self.max_guesses = max_guesses
        self.gamma = gamma
        self.lam = lam
        self.m = m
        self.reward_blend_factor = reward_blend_factor
        self.value_blend_factor = value_blend_factor
        self.fp_dtype = fp_dtype


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
        self.correct_cand_coeff = self.cfg.correct_cand_coeff
        self.max_guesses = self.cfg.max_guesses
        self.gamma = self.cfg.gamma
        self.lam = self.cfg.lam
        self.m = self.cfg.m
        self.reward_blend_factor = self.cfg.reward_blend_factor
        self.value_blend_factor = self.cfg.value_blend_factor
        self.fp_dtype = getattr(torch, self.cfg.fp_dtype)

        # Instantiate
        self.set_loader(self.cfg.loader_cfg)

        # Aliases
        self.G = self.max_guesses
        self.R_c = self.correct_reward
        self.c = self.correct_cand_coeff

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
        guess_idx = actions["guess_idx"]                                                    # [B, *]

        # Calculate
        new_target_mask, H_new = self._calculate_entropy(data, states)                      # [B, *, T] | [B, *]
        entropy_reward = (H_old - H_new)/H_max                                              # Scales episodic rewards to [0, 1] invariant of vocab size
        correct_cand = total_target_mask.gather(-1, guess_idx.unsqueeze(-1)).squeeze(-1)    # [B, *]
        count_cand = target_mask.sum(dim=-1)                                                # [B, *]
        correct = (idx == guess_idx).float()                                                # [B, *]
        correct_reward = (                                                                  # [B, *]
            self.R_c * (((1-self.c) * correct) + (self.c/count_cand * correct_cand))
        )
        rewards = entropy_reward + correct_reward                                           # [B, *]
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

        # old_states = states.copy()  # For debugging
        # target_idx = data["total"]["batch_idx"]
        # target_tensor = data["total"]["tensor"][target_idx][:, 0]
        # target_states = data["total"]["states"][target_idx][:, 0]
        # target_words = tensor_to_words(target_tensor)
        # guess_idx = actions["guess_idx"]
        # guess_tensor = data["total"]["tensor"][guess_idx][:, 0]
        # states_alphabet = states["alphabet"][:, 0]
        # guess_words = tensor_to_words(guess_tensor)
        # print("Target:", target_words)
        # print("Guess:", guess_words)

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
            (states["t"] >= self.max_guesses),                                      # [B, *]
            torch.zeros_like(states["active_mask"], dtype=torch.bool),              # [B, *]
            states["active_mask"]                                                   # [B, *]
        )
        ## Inactive if was correct (a.k.a. won)
        active = active & (~correct.bool())                                         # [B, *]
        states["active_mask"] = active                                              # [B, *]
        states["target_mask"] = new_target_mask                                     # [B, *, T]
        states["total_target_mask"] = new_total_target_mask                         # [B, *, V]
        states["entropy"] = new_entropy                                             # [B, *]

        faulty_mask = (states["target_mask"].sum(dim=-1) == 0)
        if faulty_mask.any():
            states["alphabet"] = self._update_alphabet(data, old_states, actions)           # [B, *, 26, 11]
            print("\n")


        # Collect responses
        responses = {
            "values": values,                                                       # [B, *]
            "rewards": rewards,                                                     # [B, *]
            "expected_values": cand_values.mean(dim=-1),                            # [B, *]
            "expected_rewards": cand_rewards.mean(dim=-1),                          # [B, *]
            "correct": correct                                                      # [B, *]
        }

        return states, responses
    
    def collect_episodes_mb(self, model, data, alpha, temperature, argmax=False):
        """We keep the entire episodes on CPU, then states, actions, responses on Model device"""
        with torch.no_grad():
            # Setup
            G = self.max_guesses
            B = data["target"]["batch_idx"].shape[0]  # Could be different for last batch in the epoch
            
            # Initialize history
            episodes = move_to({"states": {}, "actions": {}, "responses": {}}, "cpu")
            states = self._init_states(batch_size=B, device=model.device)
            self._append_dict(episodes["states"], move_to(states, "cpu"))

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
        for data in tqdm(self.loader, desc=desc, leave=False):
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
        active_mask = states["active_mask"]
        actions = episodes["actions"]
        valid_mask = actions["valid_mask"]
        responses = episodes["responses"]
        rewards = responses["rewards"]
        expected_rewards = responses["expected_rewards"]
        expected_values = responses["expected_values"]
        
        # Shapes
        B, G, *dims = active_mask.shape
        G = self.max_guesses

        # Sample (we must use these values to propagate gradients through the model)
        logits, naive_values = model(states)                                    # [B, G+1, *] | [B, G+1, *]
        logits = logits[:, :-1, ...]                                            # [B, G, *]
        probs = model._make_probs(logits, alpha, temperature, valid_mask)       # dict of [B, G, *, V]

        # Mask the tensors
        # NOTE: Important for GAE to not be influenced by values/rewards after an episode finishes!
        values = (naive_values * active_mask.float())                           # [B, G+1, *]
        rewards = rewards * active_mask[:, :-1, ...].float()                    # [B, G, *]
        expected_values = (                                                     # [B, G, *]
            expected_values * active_mask[:, 1:, ...].float()
        ).detach()
        expected_rewards = (                                                                # [B, G, *]
            expected_rewards * active_mask[:, :-1, ...].float()
        ).detach()

        # Compute advantages
        advantages = torch.zeros((B, G, *dims), dtype=self.fp_dtype, device=device)         # [B, G, *]
        gae = torch.zeros((B, *dims), dtype=self.fp_dtype, device=device)                   # [B, *]
        for t in reversed(range(G)):
            blended_rewards = (                                                             # [B, G, *]
                self.reward_blend_factor * expected_rewards[:, t, ...] + 
                (1 - self.reward_blend_factor) * rewards[:, t, ...]
            )
            blended_values = (                                                              # [B, G, *]
                self.value_blend_factor * expected_values[:, t, ...] + 
                (1 - self.value_blend_factor) * values[:, t+1, ...]
            ).detach()  # this is used for the values target so should not propagate gradients
            delta = blended_rewards + self.gamma * blended_values - values[:, t, ...]       # [B, G, *]
            gae = delta + self.gamma * self.lam * gae
            advantages[:, t, ...] = gae

        # Collect new responses
        responses["values"] = values                                                        # [B, G+1, *]
        responses["advantages"] = advantages                                                # [B, G, *]
        return probs, responses
