import numpy as np


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
        print(f"Parameter: {name}")
        for p, value in zip(percentiles, grad_percentiles):
            print(f"  {p}th percentile value: {value:.6f}")
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
        print(f"Parameter: {name}")
        for p, value in zip(percentiles, param_percentiles):
            print(f"  {p}th percentile value: {value:.6f}")
        print()
