# General
import math

# Torch
import torch

# Wordle
from wordle.data import tensor_to_words
from wordle.environment import sample_possible_targets, make_probs, select_actions, simulate_actions, normalize_probs
from wordle.utils import expand_var


def collect_episodes(
    actor_critic_net,
    total_vocab,
    target_vocab,
    total_vocab_tensor,
    target_vocab_tensor,
    total_vocab_states,
    target_vocab_states,
    selected_idx,
    target_tensor,
    alpha,
    temperature,
    max_guesses,
    correct_reward=0.1,
    m=3,
    argmax=False,
):
    """
    Collect a single Wordle episode with a *fixed length* rollout of 6 guesses.
    This will be masked to the actual number of guesses taken.

    Inputs:
    - total_vocab: the total vocabulary (10657 non-target plus 2316 target)
    - target_vocab: the vocabulary of targets (2316)
    - total_vocab_tensor: [total_vocab_size, 5]
    - target_vocab_tensor: [target_vocab_size, 5]
    - total_vocab_states: [total_vocab_size, 26, 11]
    - target_vocab_states: [target_vocab_size, 26, 11]
    - selected_idx: the index of the selected guess in the guess vocabulary
    - target_tensor: [batch_size, *, 5] the target word tensor
    - alpha, temperature: random exploration parameters
    - max_guesses: the maximum number of guesses (6 for Wordle)
    - correct_reward: the reward for a correct guess
    - m: number of candidates to sample for rewards

    Returns:
    - alphabet_states_batch: Tensor of [batch_size, *, max_guesses+1, 26, 11]
    - guess_states_batch: Tensor of [batch_size, *, max_guesses+1, max_guesses]
    - probs_batch: Tensor of [batch_size, *, max_guesses, total_vocab_size]
    - expected_values_batch: Tensor of [batch_size, *, max_guesses]
    - expected_rewards_batch: Tensor of [batch_size, *, max_guesses]
    - rewards_batch: Tensor of [batch_size, *, max_guesses]
    - guess_mask_batch: Tensor of [batch_size, *, max_guesses, total_vocab_size]
    - active_mask_batch: Tensor of [batch_size, *, max_guesses+1] (1 = valid step, 0 = after done)
    - valid_mask_batch: Tensor of [batch_size, *, max_guesses, total_vocab_size] (1 = valid action, 0 = invalid action)
    """
    with torch.no_grad():
        # Setup
        device = actor_critic_net.device
        episodes_shape = target_tensor.shape[:-1]
        total_vocab_size = len(total_vocab)  # action size (10657 + 2316)
        target_size = len(target_vocab)  # target size (2316)

        # Initialize Tensors
        alphabet_states_batch = torch.zeros(
            (*episodes_shape, max_guesses + 1, 26, 11),
            dtype=torch.float32,
            device=device,
        )  # [batch_size, *, max_guesses+1, 26, 11]
        guess_states_batch = torch.zeros(
            (*episodes_shape, max_guesses + 1, max_guesses),
            dtype=torch.float32,
            device=device,
        )  # [batch_size, *, max_guesses+1, max_guesses]
        probs_batch = torch.zeros((*episodes_shape, max_guesses, total_vocab_size), dtype=torch.float32, device=device)
        expected_values_batch = torch.zeros((*episodes_shape, max_guesses), dtype=torch.float32, device=device)
        expected_rewards_batch = torch.zeros((*episodes_shape, max_guesses), dtype=torch.float32, device=device)
        rewards_batch = torch.zeros((*episodes_shape, max_guesses), dtype=torch.float32, device=device)
        guess_mask_batch = torch.zeros((*episodes_shape, max_guesses, total_vocab_size), dtype=torch.bool, device=device)
        active_mask_batch = torch.ones((*episodes_shape, max_guesses + 1), dtype=torch.bool, device=device)
        valid_mask_batch = torch.ones((*episodes_shape, max_guesses, total_vocab_size), dtype=torch.bool, device=device)

        # Initialize environment state
        alphabet_states = torch.zeros((*episodes_shape, 26, 11), dtype=torch.float32).to(device)
        alphabet_entropy = torch.full(episodes_shape, math.log2(max(total_vocab_size, 1)), device=device, dtype=torch.float32)
        target_mask = torch.ones((*episodes_shape, target_size), dtype=torch.bool).to(device)
        guess_states = torch.zeros((*episodes_shape, max_guesses), dtype=torch.float32, device=device)
        guess_states[..., 0] = 1.0

        # Roll out up to max_guesses
        active = torch.ones(episodes_shape, dtype=torch.bool, device=device)
        for t in range(max_guesses):
            # Construct the state
            alphabet_states_batch[..., t, :, :] = alphabet_states
            guess_states_batch[..., t, :] = guess_states

            # Select actions
            probs, valid_mask, guess_idx, guess_idx_onehot, guess_tensor = select_actions(
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
                last_guess=(t==max_guesses-1),
                argmax=argmax,
            )

            # Get rewards and values for the possible target words
            m_cur = min(int(target_mask.sum(dim=-1).max().item()), m)  # sample up to m targets, but not more than available
            cand_targets = sample_possible_targets(target_mask, m_cur)  # [batch_size, m]
            cand_target_tensor = target_vocab_tensor[cand_targets]  # [batch_size, m, 5]
            _, _, _, _, cand_values, cand_rewards, _ = simulate_actions(
                actor_critic_net,
                expand_var(alphabet_states, dim=-3, size=m_cur),  # [batch_size, *, m_cur, 26, 11]
                expand_var(guess_states, dim=-2, size=m_cur),  # [batch_size, *, m_cur, max_guesses]
                expand_var(alphabet_entropy, dim=-1, size=m_cur),  # [batch_size, *, m_cur]
                expand_var(target_mask, dim=-2, size=m_cur),  # [batch_size, *, m_cur, target_vocab_size]
                expand_var(target_vocab_states, dim=0, size=m_cur),  # [m_cur, target_vocab_size, 26, 11]
                expand_var(guess_tensor, dim=-2, size=m_cur),  # [batch_size, *, m_cur, 5]
                cand_target_tensor,
                correct_reward,
            )  # [batch_size, *, m_cur], [batch_size, *, m_cur]
            expected_values = cand_values.mean(dim=-1)  # [batch_size, *]
            expected_rewards = cand_rewards.mean(dim=-1)  # [batch_size, *]

            # Update the state based on the true target word
            alphabet_states, guess_states, alphabet_entropy, target_mask, values, rewards, correct = simulate_actions(
                actor_critic_net,
                alphabet_states,
                guess_states,
                alphabet_entropy,
                target_mask,
                target_vocab_states,
                guess_tensor,
                target_tensor,
                correct_reward,
            )
            active = active & (~correct)

            # Store
            probs_batch[..., t, :] = probs
            expected_values_batch[..., t] = expected_values
            expected_rewards_batch[..., t] = expected_rewards
            rewards_batch[..., t] = rewards
            guess_mask_batch[..., t, :] = guess_idx_onehot
            active_mask_batch[..., t + 1] = active
            valid_mask_batch[..., t, :] = valid_mask

        active_mask_batch[..., -1] = 0  # Not active after last guess

        return (
            alphabet_states_batch,
            guess_states_batch,
            probs_batch,
            ~active,
            expected_values_batch,
            expected_rewards_batch,
            rewards_batch,
            guess_mask_batch,
            active_mask_batch,
            valid_mask_batch,
        )


def process_episodes(
    actor_critic_net,
    alphabet_states_batch,
    guess_states_batch,
    expected_values_batch,
    expected_rewards_batch,
    rewards_batch,
    active_mask_batch,
    valid_mask_batch,
    alpha,
    temperature,
    gamma,
    lam,
    reward_blend_factor,
    value_blend_factor,
):
    """
    Process the collected episodes to compute advantages and final probabilities.

    Inputs:
    - alphabet_states_batch: [batch_size, *, max_guesses+1, 26, 11]
    - guess_states_batch: [batch_size, *, max_guesses+1, max_guesses]
    - expected_values_batch: [batch_size, *, max_guesses]
    - expected_rewards_batch: [batch_size, *, max_guesses]
    - rewards_batch: [batch_size, *, max_guesses]
    - active_mask_batch: [batch_size, *, max_guesses+1]
    - valid_mask_batch: [batch_size, *, max_guesses, total_vocab_size]
    - alpha, temperature: exploration parameters
    - gamma, lam: reward smoothing parameters

    Returns:
    - advantages_batch: [batch_size, *, max_guesses]
    - probs_batch: [batch_size, *, max_guesses, total_vocab_size
    - correct: [batch_size, *]
    """
    # Setup
    episodes_shape = alphabet_states_batch.shape[:-3]  # [batch_size, *]
    max_guesses = alphabet_states_batch.size(-3) - 1  # max guesses (6 for Wordle)
    device = actor_critic_net.device

    # Forward to propagate gradients
    naive_logits, naive_values_batch = actor_critic_net(
        torch.cat([alphabet_states_batch.flatten(start_dim=-2, end_dim=-1), guess_states_batch], dim=-1)
    )  # shape: logits=[batch_size, *, max_guesses+1, total_vocab_size], value=[batch_size, *, max_guesses+1, 1]

    # Mask the tensors (masking values and rewards is important for the GAE to not carry out after an episode finishes)
    logits_batch = naive_logits[..., :-1, :]  # [batch_size, *, max_guesses, total_vocab_size]
    values_batch = (naive_values_batch.squeeze(-1) * active_mask_batch.float())  # [batch_size, *, max_guesses+1]
    rewards_batch = rewards_batch * active_mask_batch[..., :-1].float()  # [batch_size, *, max_guesses]
    expected_values_batch = (expected_values_batch * active_mask_batch[..., 1:].float()).detach()  # [batch_size, *, max_guesses]
    expected_rewards_batch = (expected_rewards_batch * active_mask_batch[..., :-1].float()).detach()  # [batch_size, *, max_guesses]

    # Compute advantages for batch with uniform episode size
    advantages_batch = torch.zeros((*episodes_shape, max_guesses), dtype=torch.float32, device=device)  # [batch_size, *, max_guesses]
    gae = torch.zeros(episodes_shape, dtype=torch.float32, device=device)  # [batch_size, *]
    for t in reversed(range(max_guesses)):
        blended_rewards = reward_blend_factor*expected_rewards_batch[..., t] + (1-reward_blend_factor)*rewards_batch[..., t]
        blended_values = value_blend_factor*expected_values_batch[..., t] + (1-value_blend_factor)*values_batch[..., t+1]
        delta = blended_rewards + gamma * blended_values - values_batch[..., t]
        gae = delta + gamma * lam * gae
        advantages_batch[..., t] = gae

    # Calculate the probabilities given a policy
    policy_probs_batch, final_probs_batch = make_probs(logits_batch, alpha, temperature, valid_mask_batch)  # [batch_size, *, max_guesses, total_vocab_size], [batch_size, *, total_vocab_size]
    guide_probs_batch = (policy_probs_batch.clone() * valid_mask_batch.float())  # [batch_size, *, max_guesses, total_vocab_size]
    guide_probs_batch = normalize_probs(guide_probs_batch, valid_mask_batch)

    return advantages_batch, policy_probs_batch, guide_probs_batch
