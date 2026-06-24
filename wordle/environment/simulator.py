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
            correct_blend_factor: float = 1.0,
            max_guesses: int = 6,
            m: int = 3,
            num_search_actions: int = 10,
            search_policy_mix: float = 0.2,
            fp_dtype: str = 'float32',
        ):
        self.loader_cfg = loader_cfg if loader_cfg is not None else WordleLoaderConfig()
        if isinstance(self.loader_cfg, dict):
            self.loader_cfg = WordleLoaderConfig(**self.loader_cfg)
        self.correct_reward = correct_reward
        self.correct_blend_factor = correct_blend_factor
        self.max_guesses = max_guesses
        self.m = m
        self.num_search_actions = num_search_actions
        self.search_policy_mix = search_policy_mix
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
        self.correct_blend_factor = self.cfg.correct_blend_factor
        self.max_guesses = self.cfg.max_guesses
        self.m = self.cfg.m
        self.num_search_actions = self.cfg.num_search_actions
        self.search_policy_mix = self.cfg.search_policy_mix
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
    
    def step(self, model, data, states, actions, calculate_reward=True):
        """
        Simulate one Wordle step for the actual hidden target in ``data``.

        This intentionally does not expand over sampled target candidates. Search
        rollouts already expand states over [top-k actions, sampled hidden targets],
        so doing another m-way expansion here would multiply memory by m again.
        """
        # Setup
        T = data["target"]["size"]
        V = data["total"]["size"]

        # Update what we can in states before simulating response
        # (t, alphabet, guessed_mask, active_mask)
        states["t"] = states["t"] + 1
        states["alphabet"] = self._update_alphabet(data, states, actions)           # [B, *, 26, 11]
        states["last_guess"] = (states["t"] == self.max_guesses)                    # [B, *]
        states["guessed_mask"] = states["guessed_mask"] | F.one_hot(                # [B, *, V]
            actions["guess_idx"], num_classes=V
        ).bool()

        # Calculate transition. Rewards are only needed for search rollouts; episode
        # collection only needs the updated target mask and correctness for state
        # progression/statistics.
        rewards, new_entropy, new_target_mask, correct = (                          # [B, *], [B, *], [B, *, T]
            self._calculate_rewards(data, states, actions)
        )
        if not calculate_reward:
            rewards = torch.zeros_like(rewards)

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
            "rewards": rewards,                                                     # [B, *]
            "correct": correct                                                      # [B, *]
        }

        return states, responses
    
    def _actions_from_guess_idx(self, base_actions, guess_idx):
        # Rollout steps only need the selected word indices. Avoid expanding full
        # probability tensors to [batch, repeats, top-k, m, vocab], which is the
        # main avoidable OOM risk in search target construction.
        V = base_actions["policy_probs"].shape[-1]
        return {
            "guess_idx": guess_idx,
            "guess_mask": F.one_hot(guess_idx, num_classes=V).bool(),
        }

    def _sample_action_indices(self, probs, k):
        """
        Sample k proposed rollout actions from the policy distribution.

        This is intentionally probabilistic rather than deterministic top-k:
        high-probability actions are more likely to be searched, but lower-ranked
        actions can still enter the rollout set.
        """
        *base_shape, V = probs.shape
        flat_probs = probs.reshape(-1, V)
        flat_idx = []
        for row in flat_probs:
            # Last-guess / <=2-candidate masks can leave fewer than k non-zero
            # entries. Use replacement only for those degenerate rows; they are
            # later overwritten by the normalized target-mask target anyway.
            replacement = (row > 0).sum().item() < k
            flat_idx.append(torch.multinomial(row, k, replacement=replacement))
        return torch.stack(flat_idx, dim=0).reshape(*base_shape, k)

    def _rollout_scores(self, model, data, states, base_actions, topk_idx, hidden_idx):
        """
        Score each proposed first action by averaging full-rollout reward over
        uniformly sampled feasible hidden targets. Shapes are [B, K, M].
        """
        *base_shape, K = topk_idx.shape
        M = hidden_idx.shape[-1]
        device = states["t"].device
        if tuple(states["t"].shape) != tuple(base_shape):
            raise ValueError(f"Expected top-k base shape {tuple(states['t'].shape)}, got {tuple(base_shape)}")
        if tuple(hidden_idx.shape[:-1]) != tuple(base_shape):
            raise ValueError(f"Expected hidden sample base shape {tuple(base_shape)}, got {tuple(hidden_idx.shape[:-1])}")
        rollout_states = {
            "t": expand_var(expand_var(states["t"], -1, K), -1, M),
            "last_guess": expand_var(expand_var(states["last_guess"], -1, K), -1, M),
            "alphabet": expand_var(expand_var(states["alphabet"], -3, K), -3, M),
            "active_mask": expand_var(expand_var(states["active_mask"], -1, K), -1, M),
            "guessed_mask": expand_var(expand_var(states["guessed_mask"], -2, K), -2, M),
            "target_mask": expand_var(expand_var(states["target_mask"], -2, K), -2, M),
            "total_target_mask": expand_var(expand_var(states["total_target_mask"], -2, K), -2, M),
            "entropy": expand_var(expand_var(states["entropy"], -1, K), -1, M),
            "idx": expand_var(hidden_idx, -2, K),
        }
        rollout_data = self.loader._idx2data(rollout_states["idx"])
        rollout_data = move_to(rollout_data, data["total"]["tensor"].device)
        first_guess_idx = topk_idx.unsqueeze(-1).expand(*base_shape, K, M)
        if tuple(first_guess_idx.shape) != (*base_shape, K, M):
            raise ValueError(f"Unexpected first action expansion shape: {tuple(first_guess_idx.shape)}")
        rollout_actions = self._actions_from_guess_idx(base_actions, first_guess_idx)

        scores = torch.zeros((*base_shape, K, M), dtype=self.fp_dtype, device=device)
        rollout_states, responses = self.step(model, rollout_data, rollout_states, rollout_actions)
        scores = scores + responses["rewards"]

        for _ in range(self.max_guesses - 1):
            if not rollout_states["active_mask"].any():
                break
            rollout_actions = model.sample(rollout_states, alpha=0.0, temperature=1.0, argmax=True)
            rollout_states, responses = self.step(model, rollout_data, rollout_states, rollout_actions)
            scores = scores + responses["rewards"]
        return scores.mean(dim=-1)

    def search_actions(self, model, data, states, alpha, temperature, argmax=False):
        """
        Refine the model policy with policy-guided complete rollouts.

        k model actions are sampled from the alpha/temperature-adjusted policy,
        then scored against m uniformly sampled feasible hidden targets. Every
        action after the proposed first action is chosen by a fully deterministic
        argmax policy. Rollout scores are standardized before softmaxing. Only
        the probability mass already assigned to the sampled actions is
        redistributed according to rollout score; all non-selected action
        probabilities are left unchanged. The search-refined distribution is
        then mixed into the prior policy with ``search_policy_mix``. For states
        with <=2 feasible targets, the normalized target mask is used directly
        as the target.
        """
        base_actions = model.sample(states, alpha, temperature, argmax=False)
        probs = base_actions["policy_probs_masked"]
        target_count = states["target_mask"].sum(dim=-1)
        small_mask = target_count <= 2

        k = min(self.num_search_actions, probs.shape[-1])
        topk_idx = self._sample_action_indices(probs, k)
        hidden_idx = self._sample_targets(states["target_mask"])
        scores = self._rollout_scores(model, data, states, base_actions, topk_idx, hidden_idx)

        score_mean = scores.mean(dim=-1, keepdim=True)
        score_std = scores.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        normalized_scores = (scores - score_mean) / score_std

        search_probs = probs.clone()
        selected_mass = probs.gather(-1, topk_idx).sum(dim=-1, keepdim=True)
        topk_probs = F.softmax(normalized_scores, dim=-1) * selected_mass
        search_probs.scatter_(-1, topk_idx, topk_probs)

        mixed_search_probs = (
            self.search_policy_mix * search_probs +
            (1 - self.search_policy_mix) * probs
        )

        target_probs = torch.where(
            small_mask.unsqueeze(-1),
            states["total_target_mask"].float() / states["total_target_mask"].float().sum(dim=-1, keepdim=True).clamp_min(self.eps),
            mixed_search_probs,
        )
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        if argmax:
            guess_idx = target_probs.argmax(dim=-1)
        else:
            *dims, V = target_probs.shape
            guess_idx = torch.multinomial(target_probs.reshape(-1, V), 1).squeeze(-1).reshape(*dims)
        base_actions["guess_idx"] = guess_idx
        base_actions["guess_mask"] = F.one_hot(guess_idx, num_classes=target_probs.shape[-1]).bool()
        base_actions["search_target_probs"] = target_probs
        return base_actions
    
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
                # Select collection actions directly from the model. Search targets are built later
                # when processing the collected states for training/evaluation loss.
                actions = model.sample(states, alpha, temperature, argmax=argmax)

                # Simulate responses = f(states, actions)
                states, responses = self.step(model, data, states, actions, calculate_reward=False)

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

    def _slice_states_at_turn(self, states, turn):
        return {k: v[:, turn, ...] if isinstance(v, torch.Tensor) else v for k, v in states.items()}

    def build_search_targets_for_states(self, model, states, alpha, temperature):
        """Build search-distillation targets for already-collected episode states."""
        targets = []
        G = self.max_guesses
        with torch.no_grad():
            for turn in range(G):
                turn_states = self._slice_states_at_turn(states, turn)
                turn_data = self.loader._idx2data(turn_states["idx"])
                turn_data = move_to(turn_data, turn_states["t"].device)
                search_actions = self.search_actions(
                    model,
                    turn_data,
                    turn_states,
                    alpha,
                    temperature,
                    argmax=True,
                )
                targets.append(search_actions["search_target_probs"])
        return torch.stack(targets, dim=1)

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
        responses["search_target_probs"] = self.build_search_targets_for_states(
            model, states, alpha, temperature
        )
        return probs, responses
