# General
import os
from tqdm import tqdm

# Torch
import torch

# Wordle
from wordle.environment import update_alphabet_states


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
    #         simulated_alphabet_state, _ = update_alphabet_states(alphabet_state.unsqueeze(0), total_vocab_tensor[i].unsqueeze(0), target_vocab_tensor[j].unsqueeze(0))
    #         simulated_entropy = torch.log2((simulated_alphabet_state <= target_vocab_states).all(dim=-1).all(dim=-1).sum(dim=-1).squeeze())
    #         possible_entropies[i, j] = simulated_entropy

    # Update states in single for-loop (higher memory requirement)
    possible_entropies = torch.zeros([total_vocab_size, masked_target_vocab_size], dtype=torch.float32, device=device)
    for i in range(total_vocab_size):
        simulated_alphabet_state, _ = update_alphabet_states(
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
    new_alphabet_state, correct = update_alphabet_states(alphabet_state.unsqueeze(0), guess_tensor.unsqueeze(0), target_tensor.unsqueeze(0))  # [26, 11]
    new_alphabet_state = new_alphabet_state.squeeze(0)  # [26, 11]
    correct = correct.squeeze(0)  # [1]

    # Get new target mask
    new_target_mask = (new_alphabet_state.unsqueeze(0) <= target_vocab_states).all(dim=-1).all(dim=-1).squeeze(0)  # [target_vocab_size]

    return new_alphabet_state, correct, new_target_mask
