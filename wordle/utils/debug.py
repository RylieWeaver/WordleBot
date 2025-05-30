# General
import numpy as np
from typing import Tuple

# Torch
import torch


def examine_gradients(actor_critic_net):
    # Observe gradient values at each 10th percentile
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    gradient_data = []

    for name, param in actor_critic_net.named_parameters():
        if param.grad is not None:
            grad_flat = param.grad.view(-1).detach().cpu().numpy()  # Flatten gradient
            grad_percentiles = np.percentile(grad_flat, percentiles)  # Compute percentiles
            gradient_data.append((name, grad_percentiles))

    # Display results
    for name, grad_percentiles in gradient_data:
        print(f'Parameter: {name}')
        for p, value in zip(percentiles, grad_percentiles):
            print(f'  {p}th percentile value: {value:.6f}')
        print()


def examine_parameters(actor_critic_net):
    # Observe parameter values at each 10th percentile
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    parameter_data = []

    for name, param in actor_critic_net.named_parameters():
        if param is not None:
            # Flatten the parameter tensor
            param_flat = param.view(-1).detach().cpu().numpy()
            param_percentiles = np.percentile(param_flat, percentiles)
            parameter_data.append((name, param_percentiles))

    # Display results
    for name, param_percentiles in parameter_data:
        print(f'Parameter: {name}')
        for p, value in zip(percentiles, param_percentiles):
            print(f'  {p}th percentile value: {value:.6f}')
        print()


def measure_grad_norms(
    actor_critic_net,
    optimizer,
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
):
    """
    Run backward passes to measure the gradient‐norm of:
        1) actor_coef * actor_loss
        2) critic_coef * critic_loss
        3) entropy_coef * entropy_loss
        4) kl_reg_coef * kl_reg_loss
        5) kl_guide_coef * kl_guide_loss
    """
    # Zero before
    optimizer.zero_grad()

    # Define grad_norm helper function
    def grad_norm(model, loss):
        loss.backward(retain_graph=True)
        norm = torch.sqrt(sum(
            (p.grad.norm()**2 for p in model.parameters() if p.grad is not None),
            torch.tensor(0.0, device=next(model.parameters()).device)
        )).item()
        optimizer.zero_grad()
        return norm

    # Compute
    actor_norm = grad_norm(actor_critic_net, actor_coef * actor_loss)
    critic_norm = grad_norm(actor_critic_net, critic_coef * critic_loss)
    entropy_norm = grad_norm(actor_critic_net, entropy_coef * entropy_loss)
    kl_reg_norm = grad_norm(actor_critic_net, kl_reg_coef * kl_reg_loss)
    kl_guide_norm = grad_norm(actor_critic_net, kl_guide_coef * kl_guide_loss)
    kl_best_norm = grad_norm(actor_critic_net, kl_best_coef * kl_best_loss)

    return actor_norm, critic_norm, entropy_norm, kl_reg_norm, kl_guide_norm, kl_best_norm
