# General
import os
import string
import numpy as np

# Torch
import torch


def get_vocab(guess_vocab_size=None, target_vocab_size=None, answers_type="updated"):
    """
    Load the vocabulary (or a subset of it) from the text files. 
    The words are loaded from two text files:
    - answer_list.txt: contains the list of possible answers
    - guess_list.txt: contains the list of possible guesses that are not in the answer list

    Inputs:
    - *_vocab_size: the number of words to subset

    Returns:
    - *_vocab: a list of words, each word is a string of length 5
    """
    # Setup
    dir = os.path.dirname(__file__)
    answers_fname = 'answer_list_plus.txt' if answers_type == "updated" else 'answer_list.txt'
    guess_vocab = []
    target_vocab = []
    
    # Guess vocab
    with open(os.path.join(dir, 'guess_list.txt')) as f:
        guess_vocab.extend(line.strip() for line in f)
    # Subset
    if guess_vocab_size is not None:
        guess_vocab = np.random.choice(guess_vocab, guess_vocab_size, replace=False)
    else:
        guess_vocab = np.array(guess_vocab)

    # Target vocab
    with open(os.path.join(dir, answers_fname)) as f:
        target_vocab.extend(line.strip() for line in f)
    # Subset
    if target_vocab_size is not None:
        target_vocab = np.random.choice(target_vocab, target_vocab_size, replace=False)
    else:
        target_vocab = np.array(target_vocab)
    
    return np.concatenate((guess_vocab, target_vocab), axis=0), target_vocab


letter_to_idx = {ch: i for i, ch in enumerate(string.ascii_lowercase)}


def words_to_tensor(words):
    """
    Convert a list of words to a tensor representation. Each letter in the word is
    represented by its index in the alphabet (0-25).

    Inputs:
    - words: list of words, each word is a string of length 5

    Returns:
    - words_tensor: [batch_size, 5]
    """
    mapped = []
    for w in words:
        mapped.append([letter_to_idx[ch] for ch in w])
    words_tensor = torch.tensor(mapped, dtype=torch.long)
    return words_tensor


idx_to_letter = {i: ch for i, ch in enumerate(string.ascii_lowercase)}


def tensor_to_words(tensor):
    """
    Convert a tensor representation of words back to a list of words.

    Inputs:
    - tensor: [batch_size, 5]

    Returns:
    - words: list of words, each word is a string of length 5
    """
    words = []
    for w in tensor:
        words.append(''.join(idx_to_letter[i.item()] for i in w))
    return words


def construct_vocab_states(vocab_tensor):
    """
    Build a tensor representation for the state of each word in 'vocab', 
    capturing all information of how letters appear (or not).

    Inputs:
    - vocab_tensor: [vocab_size, 5]

    Intermediate:
    - alphabet_tensor: 1-26 repeated 5 times -- [vocab_size, 26, 5]
    - green: boolean mask for letter placement -- [vocab_size, 26, 5]
    - count: count of letter appearances -- [vocab_size, 26, 1]
    - grey: boolean mask for NOT letter placement -- [vocab_size, 26, 5]

    Returns:
    - vocab_states: the concatenation of count, green, grey -- [vocab_size, 26, 11]
    """
    # Initialize
    device = vocab_tensor.device
    alphabet_tensor = torch.arange(0, 26).view(1, -1, 1).to(device) # [1, 26, 1]

    # Mask and count
    green = vocab_tensor.unsqueeze(1) == alphabet_tensor  # [vocab_size, 26, 5]
    count = torch.sum(green, dim=-1, keepdim=True)  # [vocab_size, 26, 1]
    grey = ~green  # [vocab_size, 26, 5]

    # Return
    vocab_states = torch.cat((count, green, grey), dim=-1).float()  # [vocab_size, 26, 11]
    return vocab_states
