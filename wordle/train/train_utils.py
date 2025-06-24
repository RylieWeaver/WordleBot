# General
import torch
import torch.nn.functional as F


def calculate_loss(
    advantages,
    old_probs,
    guide_probs,
    best_probs,
    probs,
    actor_coef,
    critic_coef,
    entropy_coef,
    kl_reg_coef,
    kl_guide_coef,
    kl_best_coef,
    guess_mask,
    active_mask,
    valid_mask,
    norm=True,
):
    """
    Calculate the loss components and total weighted loss:

    Inputs:
    - advantages: [batch_size, max_guesses]
    - old_probs: [batch_size, max_guesses, vocab_size]
    - guide_probs: [batch_size, max_guesses, vocab_size]
    - best_probs: [batch_size, max_guesses, vocab_size]
    - probs: [batch_size, max_guesses, vocab_size]
    - coefs: coefficients for the loss components
    - guess_mask: boolean mask indicating the guess made -- [batch_size, max_guesses+1, vocab_size]
    - active_mask: boolean mask indicating if a game if still active -- [batch_size, max_guesses+1]
    - valid_mask: boolean mask indicating if a guess is valid -- [batch_size, max_guesses, vocab_size]

    Intermediate:
    - Active: mask the tensors so that we don't use inactive games for our loss / parameter update
    - Guess: mask the probs for the action that was actually chosen
    - Valid: mask the probs for the actions that are valid given our inductive biases

    Returns:
    - total_loss: weighted sum of all components
    - loss_components: actor_loss, critic_loss, entropy_loss, kl_reg_loss, kl_guide_loss, kl_best_loss
    """

    # Mask
    advantages_active = advantages[active_mask[:, :-1]]
    old_probs_active = old_probs[active_mask[:, :-1]]
    guide_probs_active = guide_probs[active_mask[:, :-1]]
    best_probs_active = best_probs[active_mask[:, :-1]]
    # Mask the probs by activity for KL, then by activate guess for actor loss
    probs_active = probs[active_mask[:, :-1]]
    active_guess_mask = guess_mask[active_mask[:, :-1]]
    chosen_probs = probs_active[active_guess_mask]

    # Calculate stats
    old_log_probs_active = torch.log(old_probs_active + 1e-9)
    guide_log_probs_active = torch.log(guide_probs_active + 1e-7)
    best_log_probs_active = torch.log(best_probs_active + 1e-9)
    log_probs_active = torch.log(probs_active + 1e-9)
    chosen_log_probs = torch.log(chosen_probs + 1e-9)

    # Entropy
    entropy_probs = probs * valid_mask  # entropy regularization should not include the probabilities which we deem invalid
    entropy_probs = entropy_probs / entropy_probs.sum(-1, keepdim=True).clamp_min(1e-9)
    entropy_log_probs = torch.log(entropy_probs + 1e-9)
    entropies = -torch.sum(entropy_probs * entropy_log_probs, dim=-1)
    entropies_active = entropies[active_mask[:, :-1]]

    # Critic loss before advantage normalization
    critic_losses = advantages_active.pow(2)

    # Normalize advantages
    if norm:
        mean_adv = advantages_active.mean()
        std_adv = advantages_active.std() + 1e-9
        advantages_active = (advantages_active - mean_adv) / std_adv
    # advantages_active = torch.tanh(advantages_active / 5.0) * 5.0  # Test if this helps with stability

    # Compute losses
    actor_loss = -(advantages_active * chosen_log_probs).mean()
    critic_loss = critic_losses.mean()
    entropy_loss = -entropies_active.mean()
    kl_reg_loss = F.kl_div(old_log_probs_active, log_probs_active, reduction='batchmean', log_target=True)
    kl_guide_loss = F.kl_div(guide_log_probs_active, log_probs_active, reduction='batchmean', log_target=True)
    kl_best_loss = F.kl_div(best_log_probs_active, log_probs_active, reduction='batchmean', log_target=True)

    # Combine losses with coefficients
    total_loss = actor_coef * actor_loss + critic_coef * critic_loss + entropy_coef * entropy_loss + kl_reg_coef * kl_reg_loss + kl_guide_coef * kl_guide_loss + kl_best_coef * kl_best_loss
    return total_loss, actor_loss, critic_loss, entropy_loss, kl_reg_loss, kl_guide_loss, kl_best_loss


def evolve_policy_params(alpha, min_alpha, alpha_step, temperature, min_temperature, temperature_decay_factor):
    """
    Decrease alpha first, then temperature. Keep track of if things changed for logging
    """
    if alpha > min_alpha:
        alpha = max(min_alpha, alpha - alpha_step)
    else:
        # If alpha is already min, reduce temperature up to the min_temperature
        temperature = max(min_temperature, temperature * temperature_decay_factor)

    return alpha, temperature


def evolve_learning_params(
    optimizer,
    alpha,
    min_alpha,
    alpha_step,
    temperature,
    min_temperature,
    temperature_decay_factor,
    lr,
    min_lr,
    global_lr_decay_factor,
    lr_decay_factor=0.5,
    best_test_actor_loss=float('inf'),
    best_test_critic_loss=float('inf'),
):
    """
    Greedify the algorithm and reset the best_test_lossx. If already fully greedified, then decay once the lr.

    Returns: 
    - lr, min_lr
    - alpha, temperature
    - best_test_actor_loss, best_test_critic_loss
    """

    # Global LR decay
    new_lr = lr * global_lr_decay_factor
    min_lr = min_lr * global_lr_decay_factor

    # Greedify first
    if alpha > min_alpha or temperature > min_temperature:
        new_alpha, new_temperature = evolve_policy_params(alpha, min_alpha, alpha_step, temperature, min_temperature, temperature_decay_factor)
        # Reset best losses
        best_test_actor_loss = float('inf')
        best_test_critic_loss = float('inf')
        print(f'  -> Evolved alpha: {alpha:.2f} -> {new_alpha:.2f}, temp: {temperature:.2f} -> {new_temperature:.2f}')
    else:
        new_alpha, new_temperature = alpha, temperature
        new_lr = max(new_lr * lr_decay_factor, min_lr)  # Ensure lr is at least min_lr
        print(f'  -> Already at min_alpha={min_alpha:.2f} and min_temperature={min_temperature:.2f}')

    # Show the new LR
    for pg in optimizer.param_groups:
        pg['lr'] = new_lr
        print(f'  -> Decayed LR from {lr:.6f} to {new_lr:.6f}\n')

    # Return updated values
    return new_lr, min_lr, new_alpha, new_temperature, best_test_actor_loss, best_test_critic_loss
