# General
import torch
import torch.nn.functional as F


def log_normalize(probs, eps=1e-12, clamp=1e-12):
    probs = probs + eps  # Avoid log(0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(clamp)  # Normalize probabilities
    return torch.log(probs)


# Freeze gradients for some probs probs outside the range
def clip_grad(probs, valid_mask, min=-0.01, max=0.99):
    keep = (((probs >= min) & (probs <= max)).float())
    return probs * keep + (1 - keep) * probs.detach()


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
    grad_range=(-0.01, 0.99),
    clip_advantages=False,
    clip_eps=0.2,
):
    """
    Calculate the loss components and total weighted loss:

    Inputs:
    - advantages: [batch_size, *, max_guesses]
    - old_probs: [batch_size, *, max_guesses, vocab_size]
    - guide_probs: [batch_size, *, max_guesses, vocab_size]
    - best_probs: [batch_size, *, max_guesses, vocab_size]
    - probs: [batch_size, *, max_guesses, vocab_size]
    - coefs: coefficients for the loss components
    - guess_mask: boolean mask indicating the guess made -- [batch_size, *, max_guesses+1, vocab_size]
    - active_mask: boolean mask indicating if a game if still active -- [batch_size, *, max_guesses+1]
    - valid_mask: boolean mask indicating if a guess is valid -- [batch_size, *, max_guesses, vocab_size]

    Intermediate:
    - Active: mask the tensors so that we don't use inactive games for our loss / parameter update
    - Guess: mask the probs for the action that was actually chosen
    - Valid: mask the probs for the actions that are valid given our inductive biases

    Returns:
    - total_loss: weighted sum of all components
    - loss_components: actor_loss, critic_loss, entropy_loss, kl_reg_loss, kl_guide_loss, kl_best_loss
    """
    # Setup
    eps = 1e-10  # Small value to avoid instabilities

    # Freeze gradient for probs outside threshold (used to avoid the kl-guide from getting too much of the gradient)
    clipped_probs = clip_grad(probs, valid_mask, grad_range[0], grad_range[1])

    # Mask prob distributions
    value_advantages_active = advantages[active_mask[..., :-1]]
    old_probs_active = old_probs[active_mask[..., :-1]]
    guide_probs_active = guide_probs[active_mask[..., :-1]]
    best_probs_active = best_probs[active_mask[..., :-1]]
    probs_active = probs[active_mask[..., :-1]]
    clipped_probs_active = clipped_probs[active_mask[..., :-1]]

    # Prob distribution log terms
    old_log_probs_active = log_normalize(old_probs_active, eps=eps)
    guide_log_probs_active = log_normalize(guide_probs_active, eps=eps)
    best_log_probs_active = log_normalize(best_probs_active, eps=eps)
    log_probs_active = log_normalize(probs_active, eps=eps)
    clipped_log_probs_active = log_normalize(clipped_probs_active, eps=eps)

    # Entropy
    entropy_probs = probs * valid_mask  # entropy regularization should not include the probabilities which we deem invalid
    entropy_probs = entropy_probs / entropy_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy_log_probs = log_normalize(entropy_probs, eps=eps)
    entropies = -torch.sum(entropy_probs * entropy_log_probs, dim=-1)
    entropies_active = entropies[active_mask[..., :-1]]

    # Probs of chosen actions are used in actor loss
    active_guess_mask = guess_mask[active_mask[..., :-1]]
    chosen_log_probs = log_probs_active[active_guess_mask]
    chosen_old_log_probs = old_log_probs_active[active_guess_mask]

    # Critic loss is computed with advantages before normalization
    critic_losses = value_advantages_active.pow(2)

    # Normalize advantages per time step
    if norm:
        advantages_masked = advantages * active_mask[..., :-1]
        num_active = active_mask[..., :-1].sum(dim=0, keepdim=True).detach()
        sum_adv = (advantages_masked).sum(dim=0, keepdim=True)
        mean_adv = (sum_adv / num_active.clamp_min(eps)).detach()
        diff_adv = (advantages - mean_adv) * active_mask[..., :-1]
        std_adv = ((diff_adv.pow(2).sum(dim=0, keepdim=True) / num_active.clamp_min(eps)).sqrt()).detach().clamp_min(eps)
        zero = torch.zeros_like(mean_adv)
        one = torch.ones_like(std_adv)
        mean_adv = torch.where(num_active <= 1, zero, mean_adv)  # skip mean normalization when not enough active games
        std_adv = torch.where(num_active <= 1, one,  std_adv)  # default to 1.0 std when not enough active games
        advantages_norm = (advantages - mean_adv) / std_adv
        policy_advantages_active = advantages_norm[active_mask[..., :-1]]

    # # Normalize advantages
    # if norm:
    #     mean_adv = advantages_active.mean().detach()
    #     std_adv = advantages_active.std().detach().clamp_min(eps)
    #     advantages_active = (advantages_active - mean_adv) / std_adv

    # Actor loss
    if clip_advantages:
        ## Proximal Policy Optimization (PPO) (for training)
        ratio = torch.exp(chosen_log_probs - chosen_old_log_probs) # π_new / π_old
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        surr1 = ratio * policy_advantages_active
        surr2 = clipped * policy_advantages_active
        actor_loss = -torch.min(surr1, surr2).mean()
    else:
        ## Plain policy-gradient (for testing)
        actor_loss = -(chosen_log_probs * policy_advantages_active).mean()

    # Critic loss
    critic_loss = critic_losses.mean()
    # Entropy loss
    entropy_loss = -entropies_active.mean()
    # KL-Div losses
    kl_reg_loss = F.kl_div(old_log_probs_active, log_probs_active, reduction='batchmean', log_target=True)
    kl_guide_loss = F.kl_div(guide_log_probs_active, clipped_log_probs_active, reduction='batchmean', log_target=True)
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
