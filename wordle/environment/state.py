# General
import math

# Torch
import torch
import torch.nn.functional as F

# Wordle
from wordle.data import tensor_to_words


def update_alphabet_states(given_alphabet_states, guess_tensor, target_tensor):
    """
    Update the alphabet state in tensor operations. Specifically,

    Inputs:
    - given_alphabet_state: [batch_size, *, 26, 11]
    - guess_tensor: letters to numbers -- [batch_size, *, 5]
    - target_tensor: letters to numbers -- [batch_size, *, 5]

    Word --> Tensor:
    - alphabet_tensor: [26, 1]
    - guess_loc: mask for guessed letter locations -- [batch_size, 26, 5]
    - target_loc: mask for target letter locations -- [batch_size, 26, 5]

    Masks:
    - guessed: number of times a letter appears in our guessed word -- [batch_size, 26, 1]
    - exist: number of times a letter appears in the target word -- [batch_size, 26, 1]
    - letter_count: minimum of guessed and exist. We know a letter must appear this many times in the target! -- [batch_size, 26, 1]
    - green_mask: boolean mask for correct letter guesses AND placement -- [batch_size, 26, 5]
    - not_exist: boolean mask for guessing a letter that is not in the target anywhere -- [batch_size, 26, 5]
    - found_not: boolean mask for guessing a letter that is in the target but not in the location -- [batch_size, 26, 5]
    - grey_mask: boolean mask for incorrect letter guesses -- [batch_size, 26, 5]

    Returns:
    - new_alphabet_state: updated [batch_size, 26, 11] tensor
    """
    # Setup
    device = given_alphabet_states.device
    batch_size, *extra_dims, n, l = given_alphabet_states.shape

    # Tensorize guess/target information
    alphabet_tensor = torch.arange(26, device=device).unsqueeze(-1)  # [26, 1]
    guess_loc = guess_tensor[..., None, :] == alphabet_tensor  # [batch_size, *, 26, 5]
    target_loc = target_tensor[..., None, :] == alphabet_tensor  # [batch_size, *, 26, 5]

    # Calculate masks
    ## Counting occurrences
    guessed = guess_loc.sum(dim=-1, keepdim=True)  # [batch_size, *, 26, 1]
    exist = target_loc.sum(dim=-1, keepdim=True)  # [batch_size, *, 26, 1]
    letter_count = torch.min(guessed, exist)  # [batch_size, *, 26, 1]
    ## Checking matches
    green_mask = guess_loc & target_loc  # [batch_size, *, 26, 5]
    ## Checking non-matches
    not_exist = ((guessed > 0) & (exist == 0)).expand(*guessed.shape[:-1], 5)  # [batch_size, *, 26, 5]
    found_not = guess_loc & ~target_loc  # [batch_size, *, 26, 5]
    occupied = green_mask.any(dim=-2, keepdim=True) & ~green_mask  # [batch_size, *, 26, 5]
    grey_mask = not_exist | found_not | occupied  # [batch_size, *, 26, 5]
    ## Checking if the entire guessed word was equal to the target
    correct = (guess_tensor == target_tensor).all(dim=-1)  # [batch_size, *]

    # Combine and update
    guess_alphabet_states = torch.cat([letter_count, green_mask, grey_mask], dim=-1).float()  # [batch_size, *, 26, 11]
    new_alphabet_states = torch.max(guess_alphabet_states, given_alphabet_states)  # [batch_size, *, 26, 11]

    return new_alphabet_states, correct


def sample_possible_targets(target_mask, m):
    """
    Sample m candidates from the target mask.

    Inputs:
    - target_mask: [batch_size, vocab_size]
    - m: number of possible target words to sample for each action
    
    Returns:
    - idx: [batch_size, m]
    """
    # Setup
    batch_size, vocab_size = target_mask.shape
    device = target_mask.device

    # Check if we have enough candidates
    mask_f = target_mask.float()
    enough = mask_f.sum(dim=-1) >= m

    # Sample with replacement if we don't have enough candidates
    idx = torch.empty((batch_size, m), dtype=torch.long, device=device)
    idx[enough] = torch.multinomial(mask_f[enough], m, replacement=False)
    idx[~enough] = torch.multinomial(mask_f[~enough], m, replacement=True)

    return idx
