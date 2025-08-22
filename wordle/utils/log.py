# General
import os
import json
import time
from typing import Tuple, List

# Torch
import torch



def measure_grad_norms(
    actor_critic_net,
    actor_loss,
    actor_coef,
    critic_loss,
    critic_coef,
    entropy_loss,
    entropy_coef,
    kl_reg_loss,
    kl_reg_coef,
    kl_guide_loss,
    kl_guide_coef,
    kl_best_loss,
    kl_best_coef,
) -> Tuple[float, float, float, float, float, float]:
    """
    Run backward passes to measure the gradient‐norm of:
        1) actor_coef * actor_loss
        2) critic_coef * critic_loss
        3) entropy_coef * entropy_loss
        4) kl_reg_coef * kl_reg_loss
        5) kl_guide_coef * kl_guide_loss
        6) kl_best_coef * kl_best_loss
    """

    # define mean_grad_norm helper function
    def _mean_grad_norm(scaled_loss: torch.Tensor) -> float:
        grads: List[torch.Tensor] = torch.autograd.grad(
            scaled_loss,
            [p for p in actor_critic_net.parameters() if p.requires_grad],
            retain_graph=True,       # keep the graph alive for the upcoming backward
            create_graph=False,
            allow_unused=True,
        )
        grads = [g for g in grads if g is not None]
        if not grads:                     # safeguard (shouldn’t happen)
            return 0.0
        return torch.stack([g.norm() for g in grads]).mean().item()
    # -----------------------------------------------------------------

    actor_norm = _mean_grad_norm(actor_coef * actor_loss)
    critic_norm = _mean_grad_norm(critic_coef * critic_loss)
    entropy_norm = _mean_grad_norm(entropy_coef * entropy_loss)
    kl_reg_norm = _mean_grad_norm(kl_reg_coef * kl_reg_loss)
    kl_guide_norm = _mean_grad_norm(kl_guide_coef * kl_guide_loss)
    kl_best_norm = _mean_grad_norm(kl_best_coef * kl_best_loss)

    return actor_norm, critic_norm, entropy_norm, kl_reg_norm, kl_guide_norm, kl_best_norm


def save_checkpoint(actor_critic_net, accuracy, guesses, config, checkpoint_dir, name='model.pth'):
    # Model
    torch.save(actor_critic_net.state_dict(), f'{checkpoint_dir}/{name}')
    # Config
    with open(f'{checkpoint_dir}/config.json', 'w') as f:
        json.dump(config, f)
    # Results
    with open(f'{checkpoint_dir}/stats.json', 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'avg_guesses': guesses,
        }, f, indent=4)
    # Show
    print(f"  -> Model saved with accuracy {accuracy:.2%} and guesses {guesses:.2f}")


def clear_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        print("No GPU available, cannot clear cache.")


def rest_computer(size, multiplier=0.001):
    time.sleep(multiplier * size)
