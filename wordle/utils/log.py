# General
import os
import json
import time

# Torch
import torch


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


def rest_computer(size, multiplier=(1.0/400.0)):
    time.sleep(multiplier * size)
