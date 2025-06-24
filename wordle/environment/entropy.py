# General
import torch


def calculate_alphabet_entropy(given_target_mask, alphabet_states, vocab_states):
    """
    Inputs:
    - given_target_mask: mask of possible targets given the state -- [batch_size, vocab_size]
    - alphabet_states: [batch_size, *, 26, 11]
    - vocab_states: [*, vocab_size, 26, 11]

    Returns:
    - scores: [batch_size]
    """
    # Initialize
    device = alphabet_states.device
    vocab_states = vocab_states.unsqueeze(0)  # [1, *, vocab_size, 26, 11]
    alphabet_states = alphabet_states.unsqueeze(-3)  # [batch_size, *, 1, 26, 11]

    # Get Entropy
    new_target_mask = (alphabet_states <= vocab_states).all(dim=-1).all(dim=-1)  # [batch_size, *, vocab_size]
    # new_target_mask = (alphabet_states.cpu() <= vocab_states.cpu()).any(dim=-1).any(dim=-1).to(device)  # large tensor comparison can only be done on cpu
    entropies = torch.log2(new_target_mask.sum(dim=-1).float().clamp_min(1.0))  # [batch_size, *]  (clamp to avoid log2(0))

    # Return score
    return entropies, new_target_mask


def calculate_entropy_rewards(given_alphabet_entropy, given_target_mask, new_alphabet_states, target_vocab_states, correct, correct_multiple=1.0):
    """
    Calculate the reward based off reduction in entropy.

    Inputs:
    - given_alphabet_entropy: [batch_size, *]
    - given_target_mask: [batch_size, *, target_vocab_size]
    - new_alphabet_states: [batch_size, *, 26, 11]
    - target_vocab_states: [*, targt_vocab_size, 26, 11]
    - correct: [batch_size, *]

    Returns:
    - rewards: [batch_size, *]
    """
    # Setup
    device = new_alphabet_states.device
    target_vocab_entropy = torch.log2(torch.tensor(given_target_mask.size(-1), dtype=torch.float32).clamp_min(1.0))  # [batch_size, *] (clamp to avoid log2(0))

    # Calculate
    new_alphabet_entropy, new_target_mask = calculate_alphabet_entropy(given_target_mask, new_alphabet_states, target_vocab_states)  # [batch_size, *]
    rewards = (given_alphabet_entropy - new_alphabet_entropy)/target_vocab_entropy + (correct_multiple * correct).to(device)  # [batch_size, *]  [0,1] range for each reward part

    return rewards, new_alphabet_entropy, new_target_mask


# NOTE: Testing
def calculate_entropy_rewards_loop(given_alphabet_entropy, given_target_mask, new_alphabet_states, target_vocab_states, correct, correct_multiple=1.0):
    """
    Calculate the reward based off reduction in entropy.

    Inputs:
    - given_alphabet_entropy: [batch_size, *]
    - given_target_mask: [batch_size, *, target_vocab_size]
    - new_alphabet_states: [batch_size, *, 26, 11]
    - target_vocab_states: [*, targt_vocab_size, 26, 11]
    - correct: [batch_size, *]

    Returns:
    - rewards: [batch_size, *]
    """
    # Setup
    B = new_alphabet_states.size(0)
    device = new_alphabet_states.device
    target_vocab_entropy = torch.log2(torch.tensor(given_target_mask.size(-1), dtype=torch.float32).clamp_min(1.0))  # [batch_size, *] (clamp to avoid log2(0))

    # Calculate
    for _ in range(B):
        new_alphabet_entropy, new_target_mask = calculate_alphabet_entropy(given_target_mask[0], new_alphabet_states[0], target_vocab_states[given_target_mask[0]])  # [batch_size, *]
        rewards = (given_alphabet_entropy - new_alphabet_entropy)/target_vocab_entropy + (correct_multiple * correct).to(device)  # [batch_size, *]  [0,1] range for each reward part

    return rewards, new_alphabet_entropy, new_target_mask
