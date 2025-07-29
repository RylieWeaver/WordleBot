# General
import math

# Torch
import torch
import torch.nn.functional as F

# Wordle
from wordle.data import tensor_to_words
from wordle.environment import make_probs, inductive_biases, normalize_probs, select_actions, simulate_actions, sample_possible_targets


def make_search_probs(
    logits,
    q_values,
    search_guess_idx,
    blend_factor,
    alpha,
    temperature
):
    """
    Return a [batch_size, total_vocab_size] distribution that blends policy logits, search q_values, and random exploration:

    Inputs:
    - logits: [batch_size, total_vocab_size]
    - q_values: [batch_size, k]
    - search_guess_idx: [batch_size, k]
    - blend_factor: how much to blend the policy and search distributions
    - alpha: how much to mix in uniform exploration
    - temperature: temperature for softmax scaling

    Returns:
    - final_probs: [batch_size, total_vocab_size]
    """
    # Setup
    batch_size, total_vocab_size = logits.shape

    # Policy distribution from logits
    scaled_logits = logits / temperature   # [batch_size, total_vocab_size]
    policy_probs = F.softmax(scaled_logits, dim=-1)  # [batch_size, total_vocab_size]

    # Q distribution over the k expansions
    scaled_q = q_values / temperature  # [batch_size, k]
    q_probs = F.softmax(scaled_q, dim=-1)  # [batch_size, k]

    # Insert q_probs into the top-k action slots
    search_probs = torch.zeros_like(policy_probs)  # [batch_size, total_vocab_size]
    search_probs = search_probs.scatter(dim=-1, index=search_guess_idx, src=q_probs)  # [batch_size, total_vocab_size]

    # Blend with the original policy distribution
    mixed_probs = blend_factor * policy_probs + (1 - blend_factor) * search_probs

    # Uniform mixing
    uniform_probs = torch.ones_like(policy_probs) / total_vocab_size
    final_probs = alpha * uniform_probs + (1 - alpha) * mixed_probs

    return final_probs  # [batch_size, total_vocab_size]


def select_actions_search(
    actor_critic_net,
    alphabet_states,
    guess_states,
    alphabet_entropy,
    total_vocab,
    target_vocab,
    total_vocab_tensor,
    target_vocab_tensor,
    total_vocab_states,
    target_vocab_states,
    guess_mask_batch,
    target_mask,
    selected_idx,
    active,
    remaining_guesses,
    alpha,
    temperature,
    k=4,
    r=5,
    m=5,
    gamma=0.99,
    blend_factor=0.8,
    peek=0.0,
    argmax=False,
):
    """
    For each environment in the batch, search, estimate total rewards, compute probabilities, and select a word.

    Inputs:
    - actor_critic_net: the actor-critic network
    - alphabet_states: [batch_size, 26, 11]
    - guess_states: [batch_size, max_guesses]
    - alphabet_entropy: [batch_size]
    - total_vocab: the total vocabulary (10657 non-target plus 2315 target)
    - target_vocab: the vocabulary of targets (2315)
    - total_vocab_tensor: [total_vocab_size, 5]
    - target_vocab_tensor: [target_vocab_size, 5]
    - total_vocab_states: [total_vocab_size, 26, 11]
    - target_vocab_states: [target_vocab_size, 26, 11]
    - guess_mask_batch: [batch_size, max_guesses, total_vocab_size]
    - target_mask: [batch_size, vocab_size]
    - selected_idx: [batch_size]
    - active: [batch_size]
    - remaining_guesses: the remaining number of guesses the the wordle batch rollout
    - alpha: the uniform probability mixing coefficient
    - temperature: the temperature for softmax
    - k: number of actions to sample
    - r: number of rollouts to perform for each action
    - m: number of possible target words to sample for each action
    - gamma: discount factor for future rewards
    - blend_factor: how much to blend the policy vs Q-value
    - peek: how much probability to peek at the target word (0.0 means no peeking, 1.0 means full peeking)
    - argmax: whether to use argmax or sampling

    Returns:
    - policy_probs: [batch_size, total_vocab_size]
    - final_probs: [batch_size, total_vocab_size]
    - guess_idx: [batch_size]
    - guess_idx_onehot: [batch_size, total_vocab_size]
    - guess_tensor: [batch_size, 5]
    """
    # Setup
    device = actor_critic_net.device
    batch_size, n, l = alphabet_states.shape
    max_guesses = guess_states.shape[-1]
    total_vocab_size = len(total_vocab)
    target_vocab_size = len(target_vocab)
    total_target_mask = torch.cat([torch.zeros([batch_size, total_vocab_size - target_vocab_size], device=device, dtype=torch.bool), target_mask], dim=-1)  # [batch_size, total_vocab_size]
    active = active[:, None, None, None].expand(-1, k, r, m)  # [batch_size, k, m]
    active_mask_batch = torch.zeros(batch_size, k, r, m, remaining_guesses+1, dtype=torch.bool, device=device)
    active_mask_batch[..., 0] = active
    rewards_batch = torch.zeros(batch_size, k, r, m, remaining_guesses, dtype=torch.float32, device=device)

    # Get valid action mask based on inductive biases
    valid_action_mask = inductive_biases(guess_mask_batch, target_mask, total_target_mask, last_guess=(remaining_guesses==1))

    # Make the first guess with branching
    ## Forward pass to get logits and value
    states = torch.cat([alphabet_states.flatten(start_dim=-2, end_dim=-1), guess_states], dim=-1)  # [batch_size, 26*11 + max_guesses]
    logits, _ = actor_critic_net(states)  # shape: logits=[batch_size, total_vocab_size], value=[batch_size, 1]
    policy_probs, final_probs = make_probs(logits, alpha, temperature)  # [batch_size, total_vocab_size]
    ## Apply inductive bias masking
    policy_probs_masked = (policy_probs.clone() * valid_action_mask.float())  # [batch_size, total_vocab_size]
    ## Normalize the probabilities after masking
    policy_probs_masked = normalize_probs(policy_probs_masked, valid_action_mask)

    ## Select k actions
    if not argmax:
        policy_guess_idx = torch.multinomial(policy_probs_masked, k, replacement=False)  # [batch_size, k]
    else:
        _, policy_guess_idx = torch.topk(policy_probs_masked, k=k, dim=-1)  # [batch_size, k]
    policy_guess_idx = policy_guess_idx  # [batch_size, k]
    policy_guess_idx_onehot = F.one_hot(policy_guess_idx, num_classes=policy_probs_masked.size(-1)).bool()  # [batch_size, k, total_vocab_size]
    policy_guess_tensor = total_vocab_tensor[policy_guess_idx]  # [batch_size, k, 5]

    # Construct search states
    search_alphabet_states = alphabet_states.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, k, r, m, -1, -1)  # [batch_size, k, r, m, 26, 11]
    search_guess_states = guess_states.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, k, r, m, -1)  # [batch_size, k, r, m, max_guesses]
    search_alphabet_entropy = alphabet_entropy.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, k, r, m)  # [batch_size, k, r, m]
    search_target_vocab_states = target_vocab_states.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(k, r, m, -1, -1, -1)  # [k, r, m, target_vocab_size, 26, 11]
    search_guess_tensor = policy_guess_tensor.unsqueeze(2).unsqueeze(3).expand(-1, -1, r, m, -1)  # [batch_size, k, r, m, 5]
    search_guess_mask_batch = guess_mask_batch.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, k, r, m, -1).clone()  # [batch_size, max_guesses, k, r, m, total_vocab_size]
    search_target_mask = target_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, k, r, m, -1)  # [batch_size, k, r, m, vocab_size]
    search_targets = sample_possible_targets(target_mask, m).unsqueeze(1).unsqueeze(2).expand(-1, k, r, -1)  # [batch_size, k, r, m]
    search_target_tensor = target_vocab_tensor[search_targets]  # [batch_size, k, r, m, 5]

    # Simulate the first guess
    search_alphabet_states, search_guess_states, search_alphabet_entropy, search_target_mask, search_values, search_rewards, search_correct = simulate_actions(
        actor_critic_net,
        search_alphabet_states,
        search_guess_states,
        search_alphabet_entropy,
        search_target_mask,
        search_target_vocab_states,
        search_guess_tensor,
        search_target_tensor,
    )  # [batch_size, k, r, m, 26, 11], [batch_size, k, r, m, max_guesses], [batch_size, k, r, m], [batch_size, k, r, m, vocab_size], [batch_size, k, r, m], [batch_size, k, r, m], [batch_size, k, r, m]

    # Update
    active = active & ~search_correct
    active_mask_batch[..., 0] = active
    rewards_batch[..., 0] = search_rewards

    # Simulate actions for the rest of the rollout
    for t in range(1, remaining_guesses):
        search_probs, search_guide_probs, search_valid_action_mask, search_guess_idx, search_guess_idx_onehot, search_guess_tensor = select_actions(
            actor_critic_net,
            search_alphabet_states,
            search_guess_states,
            total_vocab,
            target_vocab,
            total_vocab_tensor,
            total_vocab_states,
            search_guess_mask_batch,
            search_target_mask,
            selected_idx,
            alpha,
            temperature,
            peek=peek,
            last_guess=(t==remaining_guesses-1),
            argmax=argmax,
        )  # [batch_size, k, r, m, total_vocab_size], [batch_size, k, r, m], [batch_size, k, r, m, total_vocab_size], [batch_size, k, r, m, 5]

        # Simulate the search actions
        search_alphabet_states, search_guess_states, search_alphabet_entropy, search_target_mask, search_values, search_rewards, search_correct = simulate_actions(
            actor_critic_net,
            search_alphabet_states,
            search_guess_states,
            search_alphabet_entropy,
            search_target_mask,
            search_target_vocab_states,
            search_guess_tensor,
            search_target_tensor,
        )  # [batch_size, k, r, m, 26, 11], [batch_size, k, r, m, max_guesses], [batch_size, k, r, m], [batch_size, k, r, m, vocab_size], [batch_size, k, r, m], [batch_size, k, r, m], [batch_size, k, r, m]

        # Update
        active = active & ~search_correct
        active_mask_batch[..., t+1] = active
        rewards_batch[..., t] = search_rewards
        search_guess_mask_batch[:, (max_guesses-remaining_guesses+t-1), ...] = search_guess_idx_onehot  # [batch_size, max_guesses+1, k, r, m, total_vocab_size]

    # Do discounted reward calculation
    active_rewards_batch = rewards_batch * active_mask_batch[..., :-1]  # [batch_size, k, r, m, remaining_guesses]
    expected_rewards_batch = active_rewards_batch.mean(dim=(-3,-2))  # [batch_size, k, remaining_guesses]
    discount = gamma ** torch.arange(remaining_guesses, device=device)  # [1, 1, remaining_guesses]
    discounted_rewards_batch = (expected_rewards_batch * discount).sum(dim=-1)  # [batch_size, k]

    # Make search probs
    q_values = discounted_rewards_batch  # [batch_size, k]
    # norm_q_values = (q_values - q_values.mean(dim=-1, keepdim=True)) / (q_values.std(dim=-1, keepdim=True) + 1e-8)  # [batch_size, k]
    final_probs = make_search_probs(
        logits,  # [batch_size, total_vocab_size]
        q_values,  # [batch_size, k]
        policy_guess_idx,  # [batch_size, k]
        blend_factor,
        alpha,
        temperature,
    )  # [batch_size, total_vocab_size]
    ## Apply inductive bias masking
    final_probs = (final_probs.clone() * valid_action_mask.float())  # [batch_size, total_vocab_size]
    ## Normalize the probabilities after masking
    final_probs = normalize_probs(final_probs, valid_action_mask)

    # Select one action based on the updated probs
    if not argmax:
        guess_idx = torch.multinomial(final_probs, 1).squeeze(-1)  # [batch_size]
    else:
        _, guess_idx = torch.topk(final_probs, k=1, dim=-1)  # [batch_size, 1]
        guess_idx = guess_idx.squeeze(-1)  # [batch_size]

    # Convert indices to words and one-hot
    guess_tensor = total_vocab_tensor[guess_idx]  # [batch_size, 5]
    guess_idx_onehot = F.one_hot(guess_idx, num_classes=logits.shape[-1]).float()  # [batch_size, total_vocab_size]

    return policy_probs, final_probs, valid_action_mask, guess_idx, guess_idx_onehot, guess_tensor
