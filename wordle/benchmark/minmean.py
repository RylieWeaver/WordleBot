# General
import os
from tqdm import tqdm

# Torch
import torch

# Wordle
from wordle.data import get_vocab, words_to_tensor, tensor_to_words, construct_vocab_states
from wordle.benchmark import get_entropy_table, simulate_action_benchmark


def select_action_minmean(alphabet_state, total_vocab_tensor, target_vocab_tensor, target_vocab_states, target_mask, first_guess=False):
    # Setup
    device = alphabet_state.device
    total_vocab_size = total_vocab_tensor.shape[0]
    target_vocab_size = target_vocab_tensor.shape[0]

    # Guess the target if it's the only thing possible. Otherwise, guess the minmean entropy
    if target_mask.sum(dim=-1) == 1:
        return (total_vocab_size - target_vocab_size) + torch.where(target_mask)[0]
    
    # # Guess the best starter word (is the same for all target words and saves time to just compute once)
    # if first_guess:
    #     return torch.tensor(113, dtype=torch.int64, device=device)  # 'aesir' is the best starter word for minmean strategy
    
    # Get the entropy table
    possible_entropies = get_entropy_table(alphabet_state, total_vocab_tensor, target_vocab_tensor, target_vocab_states, target_mask)  # [total_vocab_size, masked_target_vocab_size]

    # Find minmean guess
    mean_entropies = possible_entropies.mean(dim=-1)  # [total_vocab_size]
    guess_idx = mean_entropies.argmin(dim=-1)  # [1]

    return guess_idx  # [1]


def get_minmean_stats(total_vocab, target_vocab, checkpoint=False):
    """
    Get the statistics for the minmean strategy.
    
    Inputs:
    - total_vocab: list of possible guesses
    - target_vocab: list of possible targets
    
    Returns:
    - total_vocab_tensor: tensor representation of the total vocabulary
    - target_vocab_tensor: tensor representation of the target vocabulary
    - total_vocab_states: states for the total vocabulary
    - target_vocab_states: states for the target vocabulary
    """
    # Setup
    device = 'mps'
    max_guesses = 6
    total_correct = 0
    total_guesses = 0
    stats_dir = "minmean_stats"
    if checkpoint:
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

    # Make vocab tensors
    total_vocab_tensor = words_to_tensor(total_vocab).to(device)  # [total_vocab_size, 5]
    target_vocab_tensor = words_to_tensor(target_vocab).to(device)  # [target_vocab_size, 5]
    total_vocab_states = construct_vocab_states(words_to_tensor(total_vocab).to(device))  # [total_vocab_size, 26, 11]
    target_vocab_states = construct_vocab_states(words_to_tensor(target_vocab).to(device))  # [target_vocab_size, 26, 11]

    # Do each possible target word
    for i in tqdm(range(len(target_vocab)), desc="Evaluating minmean"):
        # Initialize
        alphabet_state = torch.zeros((26, 11), dtype=torch.float32, device=device)  # [26, 11]
        correct = torch.tensor(0)  # [1]
        target_mask = torch.ones(len(target_vocab), dtype=torch.bool, device=device)
        # Do all the guesses and stop if got correct
        guesses = torch.tensor(0)  # Guess count
        while guesses < max_guesses and not correct.bool():
            guess_idx = select_action_minmean(alphabet_state, total_vocab_tensor, target_vocab_tensor, target_vocab_states, target_mask, first_guess=((guesses==0) and (len(total_vocab)==12972)))  # Only use first guess for full vocab as well (12972 = 10657+2315)
            alphabet_state, correct, target_mask = simulate_action_benchmark(alphabet_state, total_vocab_tensor[guess_idx.squeeze()], target_vocab_tensor[i], target_vocab_states)  # [26, 11]
            guesses += 1

        # Calculate stats
        total_correct += correct  # [1]
        total_guesses += guesses  # [1]

        # Save the correct and guess to a file
        if checkpoint:
            with open(os.path.join(stats_dir, f"word_{i}.txt"), "a") as f:
                f.write(f"Target: {target_vocab[i]}, Correct: {correct.item()}, Guesses: {guesses.item()}\n")

    # Average stats
    avg_correct = total_correct.float() / float(len(target_vocab))
    avg_guesses = total_guesses.float() / float(len(target_vocab))

    return avg_correct.item(), avg_guesses.item()


if __name__ == "__main__":
    # Get vocab
    total_vocab, target_vocab = get_vocab(guess_vocab_size=1000, target_vocab_size=100)
    # total_vocab, target_vocab = get_vocab()

    # Get stats
    avg_correct, avg_guesses = get_minmean_stats(total_vocab, target_vocab, checkpoint=False)
    print(f"Average correct: {avg_correct}")
    print(f"Average guesses: {avg_guesses}")

    # Save stats in a text file
    with open("minmean_stats.txt", "w") as f:
        f.write(f"Average correct: {avg_correct}\n")
        f.write(f"Average guesses: {avg_guesses}\n")
    