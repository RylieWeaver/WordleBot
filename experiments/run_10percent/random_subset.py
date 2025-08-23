# General

# Wordle
from wordle.data import get_vocab, words_to_tensor, tensor_to_words



def main():
    # Data
    total_vocab, target_vocab = get_vocab(guess_vocab_size=1066, target_vocab_size=232)  # Load a 10% subset of the original vocab (10657 non-target plus 2315 target)
    with open('total_vocab_10percent.txt', 'w') as f:
        for word in total_vocab:
            f.write(f"{word}\n")
    with open('target_vocab_10percent.txt', 'w') as f:
        for word in target_vocab:
            f.write(f"{word}\n")

if __name__ == "__main__":
    main()
