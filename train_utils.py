import torch
import torch.nn.functional as F


def calculate_loss(
    advantages,
    old_probs,
    probs,
    actor_coef,
    critic_coef,
    entropy_coef,
    kl_coef,
    guess_mask,
    active_mask,
    norm=True,
):
    """
    Calculate the total loss for a batch given:
    (1) Advantages, log_probs, critic_losses, entropies
    (2) Masks for guess idx and when active.
    """

    # Mask
    advantages = advantages[active_mask[:, :-1]]
    old_probs = old_probs[active_mask[:, :-1]]
    # Mask the probs by activity for KL and entropy, then by activate guess for actor loss
    probs = probs[active_mask[:, :-1]]
    active_guess_mask = guess_mask[active_mask[:, :-1]]
    chosen_probs = probs[active_guess_mask]

    # Calculate stats
    old_log_probs = torch.log(old_probs + 1e-8)
    log_probs = torch.log(probs + 1e-8)
    chosen_log_probs = torch.log(chosen_probs + 1e-8)
    entropies = -torch.sum(probs * log_probs, dim=-1)
    critic_losses = advantages.pow(2)

    # Normalize advantages
    if norm:
        mean_adv = advantages.mean()
        std_adv = advantages.std() + 1e-8
        advantages = (advantages - mean_adv) / std_adv

    # Compute losses
    actor_loss = -(advantages * chosen_log_probs).mean()
    critic_loss = critic_losses.mean()
    entropy_loss = -entropies.mean()
    kl_loss = F.kl_div(old_log_probs, log_probs, reduction="batchmean", log_target=True)

    # Combine losses with coefficients
    total_loss = (
        actor_coef * actor_loss
        + critic_coef * critic_loss
        + entropy_coef * entropy_loss
        + kl_coef * kl_loss
    )
    return total_loss, actor_loss, critic_loss, entropy_loss, kl_loss


def evolve_policy_params(alpha, temperature, min_alpha, min_temperature):
    """
    Decrease alpha if above min_alpha, else decrease temperature
    if above min_temperature.
    """
    old_alpha, old_temp = alpha, temperature

    if alpha > min_alpha:
        alpha = max(min_alpha, alpha - 0.01)
    else:
        # If alpha is already min, reduce temperature
        if temperature > 3.0:
            temperature -= 0.1
        elif temperature > 1.0:
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
    init_lr,
    min_lr,
    factor=0.5,
    best_test_actor_loss=float("inf"),
    best_test_critic_loss=float("inf"),
):
    """
    1) Check if *all* param groups are already at min_lr:
       - If alpha>min_alpha or temperature>min_temperature,
         reset LR -> init_lr, evolve alpha/temperature once, reset best_test_loss.
       - Else do nothing (we're fully minimal).

    2) If *not* all at min_lr => decay for those above min_lr,
       but do NOT evolve alpha or temperature when they reach min_lr.

    Returns: (alpha, temperature, best_test_loss)
    """

    # ---------------- 1) Detect if ALL param groups are already at/below min_lr ----------------
    all_already_min = True
    for pg in optimizer.param_groups:
        if pg["lr"] > min_lr:
            all_already_min = False
            break

    if all_already_min:
        # LR is already min_lr for all groups
        if alpha > min_alpha or temperature > min_temperature:
            # => Reset LR to init_lr once, then evolve alpha/temperature
            for pg in optimizer.param_groups:
                old_lr = pg["lr"]
                if old_lr < init_lr:
                    print(
                        f"  -> LR was {old_lr:.6f}=min, resetting to init_lr={init_lr:.6f}"
                    )
                pg["lr"] = init_lr

            old_alpha, old_temp = alpha, temperature
            alpha, temperature, changed = evolve_policy_params(
                alpha, temperature, min_alpha, min_temperature
            )
            if changed:
                best_test_actor_loss = float("inf")
                best_test_critic_loss = float("inf")
                print(
                    f"  -> Evolved alpha: {old_alpha:.2f} -> {alpha:.2f}, "
                    f"temp: {old_temp:.2f} -> {temperature:.2f}.\n"
                )
        # else:
        #     # Already minimal exploration + min_lr => do nothing
        #     print("  -> All LR at min and alpha,temp minimal => no changes.\n")

        return alpha, temperature, best_test_actor_loss, best_test_critic_loss

    # ---------------- 2) Otherwise, decay any group whose LR > min_lr ----------------
    for pg in optimizer.param_groups:
        old_lr = pg["lr"]
        if old_lr > min_lr:
            new_lr = max(old_lr * factor, min_lr)
            pg["lr"] = new_lr
            if new_lr < old_lr:
                print(f"  -> Decaying LR from {old_lr:.6f} to {new_lr:.6f}\n")

    return alpha, temperature, best_test_actor_loss, best_test_critic_loss
