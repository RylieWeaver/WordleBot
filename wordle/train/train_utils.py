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
    - probs: [batch_size, max_guesses, vocab_size]
    - guess_mask: boolean mask indicating the guess made -- [batch_size, max_guesses+1, vocab_size]
    - active_mask: boolean mask indicating if a game if still active -- [batch_size, max_guesses+1]
    - valid_mask: boolean mask indicating if a guess is valid -- [batch_size, max_guesses, vocab_size]

    Intermediate:
    - Active: mask the tensors so that we don't use inactive games for our loss / parameter update
    - Guess: mask the probs for the action that was actually chosen
    - Valid: mask the probs for the actions that are valid given our inductive biases

    Returns:
    - total_loss: weighted sum of all components
    - loss_components: actor_loss, critic_loss, entropy_loss, kl_loss
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
    old_log_probs_active = torch.log(old_probs_active + 1e-8)
    guide_log_probs_active = torch.log(guide_probs_active + 1e-8)
    best_log_probs_active = torch.log(best_probs_active + 1e-8)
    log_probs_active = torch.log(probs_active + 1e-8)
    chosen_log_probs = torch.log(chosen_probs + 1e-8)

    # Entropy
    entropy_probs = probs * valid_mask
    entropy_probs = entropy_probs / entropy_probs.sum(-1, keepdim=True).clamp_min(1e-8)
    entropy_log_probs = torch.log(entropy_probs + 1e-8)
    entropies = -torch.sum(entropy_probs * entropy_log_probs, dim=-1)  # entropy regularization should not include the probabilities which we deem invalid
    entropies_active = entropies[active_mask[:, :-1]]

    # Critic loss before advantage normalization
    critic_losses = advantages_active.pow(2)

    # Normalize advantages
    if norm:
        mean_adv = advantages_active.mean()
        std_adv = advantages_active.std() + 1e-8
        advantages_active = (advantages_active - mean_adv) / std_adv

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


def evolve_policy_params(alpha, temperature, min_alpha, min_temperature, alpha_step):
    """
    Decrease alpha first, then temperature. Keep track of if things changed for logging
    """
    old_alpha, old_temp = alpha, temperature

    if alpha > min_alpha:
        alpha = max(min_alpha, alpha - alpha_step)
    else:
        # If alpha is already min, reduce temperature
        if temperature > 3.0:
            temperature -= 0.1
        elif temperature > 1.0:
            temperature -= 0.03
        elif temperature > 0.3:
            temperature -= 0.01
        else:
            temperature = max(min_temperature, temperature * 0.99)

    changed = (alpha != old_alpha) or (temperature != old_temp)
    return alpha, temperature, changed


def evolve_learning_params(
    optimizer,
    alpha,
    min_alpha,
    temperature,
    min_temperature,
    lr,
    min_lr,
    global_lr_decay,
    factor=0.5,
    best_test_actor_loss=float('inf'),
    best_test_critic_loss=float('inf'),
    alpha_step=0.01,
):
    """
    Decay the LR. If already at min_lr, reset LR, reset best_test_loss, and greedify.

    Returns: 
    - lr, min_lr
    - alpha, temperature
    - best_test_actor_loss, best_test_critic_loss
    """

    # ---------------- 1) Detect if ALL param groups are already at/below min_lr ----------------
    all_already_min = True
    for pg in optimizer.param_groups:
        if pg['lr'] > min_lr:
            all_already_min = False
            break

    if all_already_min:
        # LR is already min_lr for all groups
        if alpha > min_alpha or temperature > min_temperature:
            # => Reset LR to lr once, then evolve alpha/temperature
            for pg in optimizer.param_groups:
                old_lr = pg['lr']
                if old_lr < lr:
                    print(f'  -> LR was {old_lr:.6f}=min, resetting to lr={lr:.6f}')
                pg['lr'] = lr

            old_alpha, old_temp = alpha, temperature
            alpha, temperature, changed = evolve_policy_params(alpha, temperature, min_alpha, min_temperature, alpha_step)
            if changed:
                best_test_actor_loss = float('inf')
                best_test_critic_loss = float('inf')
                print(f'  -> Evolved alpha: {old_alpha:.2f} -> {alpha:.2f}, ' f'temp: {old_temp:.2f} -> {temperature:.2f}.\n')
        # else:
        #     # Already minimal exploration + min_lr => do nothing
        #     print("  -> All LR at min and alpha,temp minimal => no changes.\n")

        # Global LR decay
        lr = lr * global_lr_decay
        min_lr = min_lr * global_lr_decay

        return lr, min_lr, alpha, temperature, best_test_actor_loss, best_test_critic_loss

    # ---------------- 2) Otherwise, decay any group whose LR > min_lr ----------------
    for pg in optimizer.param_groups:
        old_lr = pg['lr']
        if old_lr > min_lr:
            new_lr = max(old_lr * factor, min_lr)
            pg['lr'] = new_lr
            if new_lr < old_lr:
                print(f'  -> Decaying LR from {old_lr:.6f} to {new_lr:.6f}\n')

    return lr, min_lr, alpha, temperature, best_test_actor_loss, best_test_critic_loss
