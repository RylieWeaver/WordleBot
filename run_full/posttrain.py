# General
import os

# Torch
import torch

# Wordle
from wordle.data import get_vocab, words_to_tensor, tensor_to_words
from wordle.model import ActorCriticNet, SeparatedActorCriticNet, GuessStateNet, TransformerGuessStateNet, DotGuessStateNet
from wordle.train import posttrain
from wordle.utils import load_config



def main():
    # Setup
    config = load_config('posttrain_config.json')
    load_dir = 'pretrain_log'
    checkpoint_config = load_config(f'{load_dir}/config.json')
    device = config["Model"]["device"]

    # Data
    with open('total_vocab.txt', 'r') as f:
        total_vocab = [line.strip() for line in f]
    with open('target_vocab.txt', 'r') as f:
        target_vocab = [line.strip() for line in f]

    # Model
    total_vocab_tensor = words_to_tensor(total_vocab).to(device)  # [total_vocab_size, 5]
    actor_critic_net = DotGuessStateNet(
        input_dim=checkpoint_config["Data"]["state_size"],  # 292 = 26 letters * 11 letter possibilities (1 for number, 5 green, 5 grey possibilites) plus 6 for one-hot of the number of guesses taken so far
        hidden_dim=checkpoint_config["Model"]["hidden_dim"],
        total_vocab_tensor=total_vocab_tensor,
        layers=checkpoint_config["Model"]["layers"],
        dropout=config["Model"]["dropout"],
        device=device
    ).to(device)
    actor_critic_net.load_state_dict(torch.load(f'{load_dir}/best_model.pth', map_location=device, weights_only=True))

    # Train the network
    posttrain(
        actor_critic_net,
        total_vocab,
        target_vocab,
        config["Data"]["max_guesses"],
        config["Training"]["lr"],
        config["Training"]["batch_size"],
        config["Training"]["minibatch_size"],
        config["Training"]["epochs"],
        config["Training"]["search"]["k"],
        config["Training"]["search"]["r"],
        config["Training"]["search"]["train_search"],
        config["Training"]["search"]["test_search"],
        config["Training"]["rewards"]["gamma"],
        config["Training"]["rewards"]["lam"],
        config["Training"]["rewards"]["m"],
        config["Training"]["loss"]["actor_coef"],
        config["Training"]["loss"]["critic_coef"],
        config["Training"]["loss"]["entropy_coef"],
        config["Training"]["loss"]["kl_reg_coef"],
        config["Training"]["loss"]["kl_guide_coef"],
        config["Training"]["loss"]["kl_best_coef"],
        config["Training"]["peek"],
        config["Exploration"]["alpha"],
        config["Exploration"]["min_alpha"],
        config["Exploration"]["alpha_step"],
        config["Exploration"]["temperature"],
        config["Exploration"]["min_temperature"],
        config["Exploration"]["temperature_decay_factor"],
        config["Training"]["log"]["enabled"],
        os.path.join(os.getcwd(), config["Training"]["log"]["dir"]),
        config["Training"]["scheduling"]["min_lr_factor"],
        config["Training"]["scheduling"]["global_lr_decay_factor"],
        config["Training"]["scheduling"]["lr_decay_factor"],
        config["Training"]["scheduling"]["greedify_patience"],
        config["Training"]["scheduling"]["window_size"],
        config["Training"]["scheduling"]["warmup_steps"],
        config["Training"]["scheduling"]["early_stopping_patience"],
        config
    )


if __name__ == '__main__':
    main()
