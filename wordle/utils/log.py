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


def rest_computer(target_vocab_size):
    if target_vocab_size >= 2000:
        time.sleep(20.0)
    elif target_vocab_size >= 1000:
        time.sleep(5.0)
    else:
        time.sleep(2.0)
