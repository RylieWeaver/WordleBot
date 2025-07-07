# General
import os

# Wordle
from wordle.data import get_vocab
from wordle.utils import load_config



def main():
    # Setup
    config = load_config('config.json')

    # Get Data
    total_vocab, target_vocab = get_vocab(
        target_vocab_size=config["Data"]["target_vocab_size"]
    )

    # Save Data
    with open('total_vocab_500.txt', 'w') as f:
        for word in total_vocab:
            f.write(f"{word}\n")
    with open('target_vocab_500.txt', 'w') as f:
        for word in target_vocab:
            f.write(f"{word}\n")


if __name__ == '__main__':
    main()
