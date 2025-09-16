# General
import os

# Torch
import torch

# Wordle
from wordle.data import HardWordBuffer, get_vocab, words_to_tensor, tensor_to_words
from wordle.model import ActorCriticNet, SeparatedActorCriticNet, DotGuessStateNet, DotGuessStateNet2
from wordle.train import train
from wordle.utils import load_config



# Change comments if loading from a checkpoint or fresh start
def main():
    # Setup
    config = load_config('config.json')
    device = config["Model"]["device"]
    checkpoint_config = config

    # Data
    with open('total_vocab.txt', 'r') as f:
        total_vocab = [line.strip() for line in f]
    with open('target_vocab.txt', 'r') as f:
        target_vocab = [line.strip() for line in f]

    # Run loop of model trainings
    guess_hidden_dims = [4, 8, 16, 32, 64]
    state_hidden_dims = [16 * g for g in guess_hidden_dims]
    output_dims = state_hidden_dims
    for i, (guess_hidden_dim, state_hidden_dim, output_dim) in enumerate(zip(guess_hidden_dims, state_hidden_dims, output_dims)):
        # Model
        total_vocab_tensor = words_to_tensor(total_vocab).to(device)  # [total_vocab_size, 5]
        actor_critic_net = DotGuessStateNet(
            state_input_dim=checkpoint_config["Data"]["state_size"],  # 292 = 26 letters * 11 letter possibilities (1 for number, 5 green, 5 grey possibilites) plus 6 for one-hot of the number of guesses taken so far
            state_hidden_dim=state_hidden_dim,  # varies each run
            guess_hidden_dim=guess_hidden_dim,  # varies each run
            output_dim=output_dim,  # varies each run
            total_vocab_tensor=total_vocab_tensor,
            layers=checkpoint_config["Model"]["layers"],
            dropout=config["Model"]["dropout"],
            device=device
        ).to(device)

        # Log some things
        log_dir = os.path.join(os.getcwd(), f"log_dir_{i}")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'model_size.txt'), 'a') as f:
            f.write(f"Guess hidden dim: {guess_hidden_dim}, State hidden dim: {state_hidden_dim}, Output dim: {output_dim}\n")
            num_params = sum(p.numel() for p in actor_critic_net.parameters())
            f.write(f"Number of parameters in the model: {num_params}\n")

        # Train the network
        train(
            actor_critic_net,
            total_vocab,
            target_vocab,
            config["Data"]["max_guesses"],
            config["Training"]["rollout_size"],
            config["Training"]["lr"],
            config["Training"]["clip_eps"],
            config["Training"]["max_grad_norm"],
            config["Training"]["batch_size"],
            config["Training"]["target_repeats"],
            config["Training"]["collect_minibatch_size"],
            config["Training"]["process_minibatch_size"],
            config["Training"]["epochs"],
            config["Training"]["rewards"]["correct_reward"],
            config["Training"]["rewards"]["gamma"],
            config["Training"]["rewards"]["lam"],
            config["Training"]["rewards"]["m"],
            config["Training"]["rewards"]["reward_blend_factor"],
            config["Training"]["rewards"]["value_blend_factor"],
            config["Training"]["loss"]["actor_coef"],
            config["Training"]["loss"]["critic_coef"],
            config["Training"]["loss"]["entropy_coef"],
            config["Training"]["loss"]["kl_reg_coef"],
            config["Training"]["loss"]["kl_guide_coef"],
            config["Training"]["loss"]["kl_best_coef"],
            config["Exploration"]["alpha"],
            config["Exploration"]["min_alpha"],
            config["Exploration"]["alpha_step"],
            config["Exploration"]["temperature"],
            config["Exploration"]["min_temperature"],
            config["Exploration"]["temperature_decay_factor"],
            config["Training"]["log"]["enabled"],
            log_dir,  # varies each run
            config["Training"]["scheduling"]["min_lr_factor"],
            config["Training"]["scheduling"]["global_lr_decay_factor"],
            config["Training"]["scheduling"]["lr_decay_factor"],
            config["Training"]["scheduling"]["greedify_patience"],
            config["Training"]["scheduling"]["warmup_steps"],
            config["Training"]["scheduling"]["early_stopping_patience"],
            config,
            replay_loader=None  # Using None for no replay
        )


if __name__ == '__main__':
    main()
