# General
from pathlib import Path

# Wordle
from wordle.data import get_vocab


RUN_DIR = Path(__file__).resolve().parent


def write_vocab(path, vocab):
    with path.open("w") as f:
        for word in vocab:
            f.write(f"{word}\n")


def load_data(target_vocab_size=None, nontarget_vocab_size=None, targets_type="original"):
    target_vocab = get_vocab(vocab_type="target", size=target_vocab_size, targets_type=targets_type)
    nontarget_vocab = get_vocab(vocab_type="nontarget", size=nontarget_vocab_size)
    total_vocab = list(target_vocab) + list(nontarget_vocab)
    return total_vocab, target_vocab, nontarget_vocab


def main():
    total_vocab, target_vocab, nontarget_vocab = load_data(targets_type="original")
    write_vocab(RUN_DIR / "total_vocab.txt", total_vocab)
    write_vocab(RUN_DIR / "target_vocab.txt", target_vocab)
    write_vocab(RUN_DIR / "nontarget_vocab.txt", nontarget_vocab)


if __name__ == "__main__":
    main()
