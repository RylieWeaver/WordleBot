# General
import os
import json

# Torch
import torch


def save_checkpoint(actor_critic_net, best_accuracy, best_guesses, config, checkpoint_dir):
    # Model
    torch.save(actor_critic_net.state_dict(), f'{checkpoint_dir}/model.pth')
    # Config
    with open(f'{checkpoint_dir}/config.json', 'w') as f:
        json.dump(config, f)
    # Results
    with open(f'{checkpoint_dir}/stats.json', 'w') as f:
        json.dump({
            'accuracy': best_accuracy,
            'avg_guesses': best_guesses,
        }, f, indent=4)
    # Show
    print(f"New best model saved with accuracy {best_accuracy:.2%} and guesses {best_guesses:.2f}")