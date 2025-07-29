# Torch
import torch
import torch.nn.functional as F

# Wordle
from wordle.data import tensor_to_words
from wordle.environment import update_alphabet_states, calculate_entropy_rewards


def make_probs(logits, alpha, temperature, valid_mask=None):
    """
    Returns the final probabilities after mixing with uniform distribution and temperature scaling.

    Inputs:
    - logits: [batch_size, *, total_vocab_size] tensor of logits
    - alpha: mixing coefficient for uniform distribution
    - temperature: temperature for softmax
    - valid_mask: [batch_size, *, total_vocab_size] tensor of valid actions (optional)

    Returns:
    - final_probs: [batch_size, *, total_vocab_size] tensor of probabilities
    """
    # Setup
    device = logits.device
    batch_size, *extra_dims, total_vocab_size = logits.shape
    valid_mask = valid_mask if valid_mask is not None else torch.ones_like(logits, device=device, dtype=torch.bool)  # [batch_size, *, total_vocab_size]

    # Softmax with temperature
    scaled_logits = logits / temperature
    policy_probs = F.softmax(scaled_logits, dim=-1)

    # Uniform distribution over valid actions for alpha-mixing
    uniform_probs = valid_mask.float() / valid_mask.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # [batch_size, *, total_vocab_size]
    final_probs = alpha * uniform_probs + (1 - alpha) * policy_probs

    return policy_probs, final_probs


def inductive_biases(
    guess_mask_batch,
    target_mask,
    total_target_mask,
    last_guess=False,
):
    """
    Apply inductive biases to the guess mask and target mask.

    Inputs:
    - guess_mask_batch: [batch_size, *, max_guesses+1, total_vocab_size]
    - target_mask: [batch_size, *, vocab_size]
    - total_target_mask: [batch_size, *, total_vocab_size]
    - selected_idx: [batch_size, *] indices of the target words for this batch
    - last_guess: whether this is the last guess

    Returns:
    - valid_action_mask: [batch_size, *, total_vocab_size] tensor of valid actions
    """
    # Setup
    device = guess_mask_batch.device
    valid_action_mask = torch.ones_like(total_target_mask, device=device, dtype=torch.bool)  # [batch_size, *, total_vocab_size]

    # 1) Do not do repeat guesses
    guessed = guess_mask_batch.any(dim=-2) & ~total_target_mask  # [batch_size, *, total_vocab_size]
    valid_action_mask = valid_action_mask & ~guessed  # [batch_size, *, total_vocab_size]

    # 2) Always guess a possible target word for last guess
    if last_guess:
        valid_action_mask = valid_action_mask & total_target_mask  # [batch_size, *, total_vocab_size]

    # 3) If there are two or fewer possible targets, choose from those
    idxs = torch.where((target_mask.float().sum(dim=-1) <= 2))[0]  # [batch_size, *]
    if idxs.numel() > 0:
        valid_action_mask[idxs] = valid_action_mask[idxs] & total_target_mask[idxs]  # [batch_size, *, total_vocab_size]

    return valid_action_mask


def normalize_probs(probs, valid_action_mask=None):
    """
    Normalize probabilities and apply a valid action mask.
    Inputs:
    - probs: [batch_size, *, total_vocab_size] tensor of probabilities
    - valid_action_mask: [batch_size, *, total_vocab_size] tensor of valid actions (optional)
    Returns:
    - normalized_probs: [batch_size, *, total_vocab_size] tensor of normalized probabilities
    """
    # Setup
    device = probs.device
    uniform = torch.ones_like(probs, device=device, dtype=torch.float32) / probs.shape[-1]  # [batch_size, *, total_vocab_size]
    valid_action_mask = valid_action_mask if valid_action_mask is not None else torch.ones_like(probs, device=device, dtype=torch.bool)  # [batch_size, *, total_vocab_size]

    # Set the probabilities of the zero rows (should only happen for inactive episodes, but can sometimes happen from very small values in the softmax probability)
    # option 1 is masked probs, option 2 is valid actions, and option 3 is uniform distribution
    zero_rows = (probs.sum(dim=-1) < 1e-8)  # [batch_size]
    if zero_rows.any():
        has_valid = (valid_action_mask[zero_rows].sum(dim=-1, keepdim=True) > 1e-8)  # [zero_rows, 1]
        uniform = torch.ones_like(probs[zero_rows], device=device, dtype=torch.float32) / probs[zero_rows].sum(dim=-1, keepdim=True).clamp_min(1e-9)  # [zero_rows, *, total_vocab_size]
        probs[zero_rows] = torch.where(has_valid, valid_action_mask[zero_rows].float(), uniform)  # [zero_rows, *, total_vocab_size]

    # Normalize the probabilities
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # [batch_size, *, total_vocab_size]
    while (probs.sum(-1) - 1.0).abs().max() > 1e-6:
        print("Warning: probabilities sum to more or less than 1.0")
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    return probs


def select_actions(
    actor_critic_net,
    alphabet_states,
    guess_states,
    total_vocab,
    target_vocab,
    total_vocab_tensor,
    total_vocab_states,
    guess_mask_batch,
    target_mask,
    selected_idx,
    alpha,
    temperature,
    last_guess=False,
    argmax=False,
):
    """
    For each environment in the batch, estimate values/rewards, compute probabilities, and select a word.

    Inputs:
    - actor_critic_net: the actor-critic network
    - alphabet_states: [batch_size, *, 26, 11]
    - guess_states: [batch_size, *, max_guesses]
    - total_vocab: the total vocabulary (10657 non-target plus 2315 target)
    - target_vocab: the vocabulary of targets (2315)
    - total_vocab_tensor: [total_vocab_size, 5]
    - total_vocab_states: [total_vocab_size, 26, 11]
    - target_vocab_tensor: [target_vocab_size, 5]
    - guess_mask_batch: [batch_size, *, max_guesses, total_vocab_size]
    - target_mask: [batch_size, *, vocab_size]
    - alpha: the uniform probability mixing coefficient
    - temperature: the temperature for softmax
    - selected_idx: the indices of the target words for this batch
    - peak: the probability of deterministically guessing the target word
    - argmax: whether to use argmax or sampling

    Returns:
    - policy_probs_softmask: [batch_size, *, total_vocab_size]
    - guess_idx: [batch_size, *]
    - guess_idx_onehot: [batch_size, *, total_vocab_size]
    - guess_tensor: [batch_size, *, 5]
    """

    # Setup
    device = actor_critic_net.device
    batch_size, *extra_dims, n, l = alphabet_states.shape
    total_vocab_size = len(total_vocab)
    target_vocab_size = len(target_vocab)
    total_target_mask = torch.cat([torch.zeros([batch_size, *extra_dims, total_vocab_size - target_vocab_size], device=device, dtype=torch.bool), target_mask], dim=-1)  # [batch_size, *, total_vocab_size]

    # Get valid action mask based on inductive biases
    valid_action_mask = inductive_biases(guess_mask_batch, target_mask, total_target_mask, last_guess=last_guess)

    # Forward pass to get logits and value
    states = torch.cat([alphabet_states.flatten(start_dim=-2, end_dim=-1), guess_states], dim=-1)  # [batch_size, *, 26*11 + max_guesses]
    logits, _ = actor_critic_net(states)  # shape: logits=[batch_size, *, total_vocab_size], value=[batch_size, *, 1]
    policy_probs, final_probs = make_probs(logits, alpha, temperature, valid_action_mask)  # [batch_size, *, total_vocab_size]

    # Apply inductive bias masking and normalize
    policy_probs_masked = (policy_probs.clone() * valid_action_mask.float())  # [batch_size, *, total_vocab_size]
    final_probs_masked = (final_probs.clone() * valid_action_mask.float())  # [batch_size, *, total_vocab_size]
    policy_probs_masked = normalize_probs(policy_probs_masked, valid_action_mask)
    final_probs_masked = normalize_probs(final_probs_masked, valid_action_mask)

    # Select one action based on the updated average values
    if not argmax:
        *dims, total_vocab_size = final_probs_masked.shape
        guess_idx = torch.multinomial(final_probs_masked.reshape(-1, total_vocab_size) , 1).squeeze(-1)  # [*dims]
        guess_idx = guess_idx.reshape(*dims)  # [batch_size, *]
    else:
        _, guess_idx = torch.topk(policy_probs_masked, k=1, dim=-1) # [batch_size, *, 1]
        guess_idx = guess_idx.squeeze(-1)  # [batch_size, *]

    # Convert indices to words and one-hot
    guess_tensor = total_vocab_tensor[guess_idx]  # [batch_size, *, 5]
    guess_idx_onehot = F.one_hot(guess_idx, num_classes=policy_probs_masked.shape[-1]).bool()  # [batch_size, *, total_vocab_size]

    return policy_probs, valid_action_mask, guess_idx, guess_idx_onehot, guess_tensor


def simulate_actions(
        actor_critic_net,
        alphabet_states,
        guess_states,
        alphabet_entropy,
        target_mask,
        target_vocab_states,
        guess_tensor,
        target_tensor,
        correct_reward,
    ):
    """
    Simulate one Wordle step

    Inputs:
    - alphabet_states: [batch_size, *, 26, 11]
    - guess_states: [batch_size, *, max_guesses]
    - alphabet_entropy: [batch_size, *]
    - given_target_mask: [batch_size, *, target_vocab_size]
    - target_vocab_states: [*, target_vocab_size, 26, 11]
    - guess_tensor: [batch_size, *, 5]
    - target_tensor: [batch_size, *, 5]

    Returns:
    - new_alphabet_states: updated [batch_size, *, 26, 11] tensor
    - new_guess_states: updated [batch_size, *, max_guesses] tensor
    - new_alphabet_entropy: [batch_size, *]
    - new_target_mask: [batch_size, *, target_vocab_size]
    - values: [batch_size, *]
    - rewards: [batch_size, *]
    - correct: [batch_size, *]
    """
    # Setup
    device = actor_critic_net.device
    batch_size, *extra_dims, n, l = alphabet_states.shape

    # Update states
    new_alphabet_states, correct = update_alphabet_states(alphabet_states, guess_tensor, target_tensor)  # [batch_size, *, 26, 11]
    new_guess_states = guess_states.roll(shifts=1, dims=-1)  # [batch_size, *, max_guesses]
    new_states = torch.cat([new_alphabet_states.flatten(start_dim=-2, end_dim=-1), new_guess_states], dim=-1)  # [batch_size, *, 26*11 + max_guesses]

    # Calculate candidate values
    with torch.no_grad():
        _, values = actor_critic_net(new_states)  # [batch_size, *, 1]
        values = values.squeeze(-1)  # [batch_size, *]

    # Calculate candidate rewards
    rewards, new_alphabet_entropy, new_target_mask = calculate_entropy_rewards(
        alphabet_entropy,   # [batch_size, *, 26, 11]
        target_mask,  # [batch_size, *, vocab_size]
        new_alphabet_states,  # [batch_size, *, 26, 11]
        target_vocab_states,  # [*, vocab_size, 26, 11]
        correct,  # [batch_size, *]
        correct_reward,
    )  # [batch_size, *], [batch_size, *], [batch_size, *, vocab_size]

    return new_alphabet_states, new_guess_states, new_alphabet_entropy, new_target_mask, values, rewards, correct
