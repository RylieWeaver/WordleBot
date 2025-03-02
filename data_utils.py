import numpy as np
import random


def get_vocab(vocab_size=None):
    vocab = []
    # for file in ['answer_list.txt', 'guess_list.txt']:
    for file in ['answer_list.txt']:
        with open(file) as f:
            vocab.extend(line.strip() for line in f)
    # Return subset
    if vocab_size is not None:
        return np.random.choice(vocab, vocab_size, replace=False)
    else:
        return vocab
