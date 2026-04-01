# General
import os
import string
import numpy as np
from typing import Optional, Literal

# Torch
import torch



def get_vocab(
        vocab_type: Literal["target", "nontarget"] = "target",
        size: Optional[int] = None,
        targets_type="original"
    ):
    """
    Load a vocabulary (or a subset of it) from the text files. 
    The words are loaded from one of two text files:
    - target_vocab.txt: contains the list of possible target words
    - nontarget_vocab.txt: contains the list of allowed guesses that aren't targets

    Inputs:
    - vocab_type: whether to load the target vocab or nontarget vocab
    - size: if not None, the number of words to sample from the vocab (for quick testing)
    - targets_type: whether to load the original target vocab or the updated one with more words

    Returns:
    - *_vocab: a list of words, each word is a string of length 5
    """
    # Setup
    dir = os.path.dirname(__file__)

    # Target vocab
    if vocab_type == "target":
        targets_fname = 'target_vocab_plus.txt' if targets_type == "updated" else 'target_vocab.txt'
        target_vocab = []
        with open(os.path.join(dir, targets_fname)) as f:
            target_vocab.extend(line.strip() for line in f)
        # Subset
        if size is not None:
            target_vocab = np.random.choice(target_vocab, size, replace=False)
        else:
            target_vocab = np.array(target_vocab)
        return target_vocab

    # Nontarget vocab
    if vocab_type == "nontarget":
        nontarget_vocab = []
        with open(os.path.join(dir, 'nontarget_vocab.txt')) as f:
            nontarget_vocab.extend(line.strip() for line in f)
        # Subset
        if size is not None:
            nontarget_vocab = np.random.choice(nontarget_vocab, size, replace=False)
        else:
            nontarget_vocab = np.array(nontarget_vocab)
        return nontarget_vocab
    

letter_to_idx = {ch: i for i, ch in enumerate(string.ascii_lowercase)}


def words_to_tensor(words):
    """
    Convert a list of 5-letter words to a tensor representation. Each letter
    in the word is represented by its index in the alphabet (0-25).

    Inputs:
    - words: [*]

    Returns:
    - words_tensor: [*, 5]
    """
    mapped = []
    for w in words:
        mapped.append([letter_to_idx[ch] for ch in w])
    words_tensor = torch.tensor(mapped, dtype=torch.long)
    return words_tensor


idx_to_letter = {i: ch for i, ch in enumerate(string.ascii_lowercase)}


def tensor_to_words(tensor):
    """
    Convert a tensor representation of words back to a list of 5-letter strings.

    Inputs:
    - tensor: [*, 5]

    Returns:
    - words: [*]
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
    - vocab_tensor: [*, 5]

    Intermediate:
    - alphabet_tensor: 1-26 repeated 5 times -- [*, 26, 5]
    - green: boolean mask for letter placement -- [*, 26, 5]
    - count: count of letter appearances -- [*, 26, 1]
    - grey: boolean mask for NOT letter placement -- [*, 26, 5]

    Returns:
    - vocab_states: the concatenation of count, green, grey -- [*, 26, 11]
    """
    # Initialize
    device = vocab_tensor.device
    alphabet_tensor = torch.arange(0, 26).view(1, -1, 1).to(device) # [1, 26, 1]

    # Mask and count
    green = vocab_tensor.unsqueeze(1) == alphabet_tensor    # [V, 26, 5]
    count = torch.sum(green, dim=-1, keepdim=True)          # [V, 26, 1]
    grey = ~green                                           # [V, 26, 5]

    # Return
    vocab_states = torch.cat((count, green, grey), dim=-1).float()  # [V, 26, 11]
    return vocab_states
