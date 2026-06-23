# General
import os
from tqdm import tqdm

# Torch
import torch
from torch.nn import functional as F

# Wordle



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
    target_idx = data["total"]["batch_idx"]                                     # [B, *] (the idx of the true target word in the total vocab)
    guess_idx = actions["guess_idx"]                                            # [B, *] (the idx of the guessed word in the total vocab)
    total_tensor = data["total"]["tensor"]                                      # [T, 5]
    target_tensor = total_tensor[target_idx]                                    # [B, *, 5]
    guess_tensor = total_tensor[guess_idx]                                      # [B, *, 5]
    alphabet = states["alphabet"]                                               # [B, *, 26, 11]

    # OHE guess/target information
    guess_loc = F.one_hot(guess_tensor, num_classes=26).transpose(-2, -1)       # [B, *, 5, 26] -> [B, *, 26, 5]
    target_loc = F.one_hot(target_tensor, num_classes=26).transpose(-2, -1)     # [B, *, 5, 26] -> [B, *, 26, 5]

    # Counting occurrences
    guessed = guess_loc.sum(dim=-1, keepdim=True)                               # [B, *, 26, 1]
    exist = target_loc.sum(dim=-1, keepdim=True)                                # [B, *, 26, 1]
    letter_count = torch.min(guessed, exist)                                    # [B, *, 26, 1]

    # Compute first order information (directly from guess/target)
    ## Matches
    green_mask = guess_loc & target_loc                                         # [B, *, 26, 5]
    ## Non-matches
    not_exist = ((guessed > 0) & (exist == 0)).expand_as(guess_loc)             # [B, *, 26, 5]
    found_not = guess_loc & ~target_loc                                         # [B, *, 26, 5]
    grey_mask = not_exist | found_not                                           # [B, *, 26, 5]
    ## Combine
    green_mask = green_mask | alphabet[..., 1:6].bool()                         # [B, *, 26, 5]
    grey_mask = grey_mask | alphabet[..., 6:].bool()                            # [B, *, 26, 5]

    # Compute second order information (combining known info)
    ## Green implies all others grey
    occupied = green_mask.any(dim=-2, keepdim=True) & ~green_mask               # [B, *, 26, 5]
    grey_mask = grey_mask | occupied                                            # [B, *, 26, 5]
    ## All others grey implies green
    no_others = (grey_mask.sum(dim=-2, keepdim=True) == 25)                     # [B, *, 1, 5]
    green_mask = green_mask | no_others                                         # [B, *, 26, 5]

    # Combine and update
    alphabet = torch.cat(                                                       # [B, *, 26, 11]
        [letter_count, green_mask, grey_mask], dim=-1
    ).float()

    return alphabet


def get_entropy_table(alphabet_state, total_vocab_tensor, target_vocab_tensor, target_vocab_states, target_mask):
    """
    Greedy strategy for selecting the next guess based on the maximum entropy of the target vocabulary.
    
    Inputs:
    - alphabet_state: [26, 11] tensor of alphabet state
    - total_vocab_tensor: [total_vocab_size, 5] tensor of total vocabulary
    - total_vocab_states: [total_vocab_size, 26, 11] tensor of total vocabulary states
    - target_vocab_tensor: [target_vocab_size, 5] tensor of target vocabulary
    - target_vocab_states: [target_vocab_size, 26, 11] tensor of target vocabulary states
    - target_mask: [target_vocab_size] tensor of target mask
    
    Returns:
    - guess_idx: index of the selected guess
    """
    # Setup
    device = alphabet_state.device
    total_vocab_size = total_vocab_tensor.shape[0]
    target_vocab_size = target_vocab_tensor.shape[0]

    # Apply mask
    masked_target_vocab_tensor = target_vocab_tensor[target_mask]  # shape: [possible_targets, 5]
    masked_target_vocab_states = target_vocab_states[target_mask]  # shape: [possible_targets, 26, 11]

    # Setup
    n, l = alphabet_state.shape
    total_vocab_size = total_vocab_tensor.shape[0]
    masked_target_vocab_size = masked_target_vocab_tensor.shape[0]

    # # Update states in double for-loop (lower memory requirement)
    # possible_entropies = torch.zeros([total_vocab_size, masked_target_vocab_size], dtype=torch.float32, device=device)
    # for i in range(total_vocab_size):
    #     for j in range(masked_target_vocab_size):
    #         simulated_alphabet_state, _ = _update_alphabet(alphabet_state.unsqueeze(0), total_vocab_tensor[i].unsqueeze(0), target_vocab_tensor[j].unsqueeze(0))
    #         simulated_entropy = torch.log2((simulated_alphabet_state <= target_vocab_states).all(dim=-1).all(dim=-1).sum(dim=-1).squeeze())
    #         possible_entropies[i, j] = simulated_entropy

    # Update states in single for-loop (higher memory requirement)
    possible_entropies = torch.zeros([total_vocab_size, masked_target_vocab_size], dtype=torch.float32, device=device)
    for i in range(total_vocab_size):
        simulated_alphabet_state, _ = _update_alphabet(
                                        alphabet_state.unsqueeze(0).expand(masked_target_vocab_size, -1, -1),
                                        total_vocab_tensor[i].unsqueeze(0).expand(masked_target_vocab_size, -1),
                                        masked_target_vocab_tensor
                                        )
        simulated_entropies = torch.log2((simulated_alphabet_state.unsqueeze(1).expand(-1, masked_target_vocab_size, -1, -1) <= masked_target_vocab_states.unsqueeze(0).expand(masked_target_vocab_size, -1, -1, -1)).all(dim=-1).all(dim=-1).sum(dim=-1))
        possible_entropies[i, :] = simulated_entropies

    return possible_entropies  # [total_vocab_size, masked_target_vocab_size]


def simulate_action_benchmark(
        alphabet_state,
        guess_tensor,
        target_tensor,
        target_vocab_states,
    ):
    """
    Simulate one Wordle step

    Inputs:
    - alphabet_states: [26, 11]
    - guess_tensor: [5]
    - target_tensor: [5]
    - target_vocab_states: [target_vocab_size, 26, 11]

    Returns:
    - new_alphabet_state: updated [26, 11] tensor
    - correct: [1]
    """
    # Update state
    new_alphabet_state, correct = _update_alphabet(alphabet_state.unsqueeze(0), guess_tensor.unsqueeze(0), target_tensor.unsqueeze(0))  # [26, 11]
    new_alphabet_state = new_alphabet_state.squeeze(0)  # [26, 11]
    correct = correct.squeeze(0)  # [1]

    # Get new target mask
    new_target_mask = (new_alphabet_state.unsqueeze(0) <= target_vocab_states).all(dim=-1).all(dim=-1).squeeze(0)  # [target_vocab_size]

    return new_alphabet_state, correct, new_target_mask
