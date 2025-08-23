# General

# Wordle
from wordle.data import get_vocab, words_to_tensor, tensor_to_words



def main():
    # Data
    total_vocab, target_vocab = get_vocab(answers_type="original")  # Load the full vocab (10657 non-target plus 2318 target)
    with open('total_vocab.txt', 'w') as f:
        for word in total_vocab:
            f.write(f"{word}\n")
    with open('target_vocab.txt', 'w') as f:
        for word in target_vocab:
            f.write(f"{word}\n")

if __name__ == "__main__":
    main()
