# General
import random
import numpy as np
from collections import deque, Counter

# Torch
import torch



class HardWordBuffer:
    """
    Hard-word buffer with EMA difficulty weights.
    Stores at most `capacity` unique words; each has a weight that
    is EMA-updated toward its miss-rate per episode.
    Sampling draws with replacement according to weights.
    """

    def __init__(self, vocab, capacity=100, replay_ratio=0.1, rho=0.1):
        self.vocab = vocab
        self.capacity = max(1, capacity)
        self.vocab_size = len(vocab)
        self.replay_ratio = replay_ratio
        self.rho = rho
        self.init_w = 1.0 / len(vocab)

        self.rng = random.Random()
        self.queue = deque(maxlen=self.capacity)
        self.entries = {}  # word → (idx, weight)

        # Initialize with random subset of the vocab
        initial_set = self.rng.sample(range(len(vocab)), min(capacity, len(vocab)))
        for idx in initial_set:
            word = vocab[idx]
            self.queue.append(word)
            self.entries[word] = (idx, self.init_w)

    def tensor_to_list(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.view(-1).tolist()
        else:
            return tensor

    def update(self, selected_idx, missed_idx):
        # 1) Turn tensors into lists
        selected_idx = self.tensor_to_list(selected_idx)
        missed_idx = self.tensor_to_list(missed_idx)

        # 2) Count tries & corrects
        tries = Counter(selected_idx)
        missed = Counter(missed_idx)

        # 3) EMA-update each word that was tried this episode
        for idx, total in tries.items():
            miss_rate = missed.get(idx, 0) / total
            word = self.vocab[idx]

            # Ensure membership (evict smallest if full)
            if word not in self.entries:
                if len(self.queue) == self.capacity:
                    # find & evict the word with the smallest weight
                    evict = min(self.queue, key=lambda w: self.entries[w][1])
                    self.queue.remove(evict)
                    self.entries.pop(evict, None)
                self.queue.append(word)
                self.entries[word] = (idx, self.init_w)

            # EMA toward miss_rate
            idx_old, w_old = self.entries[word]
            w_new = w_old + self.rho * (miss_rate - w_old)
            self.entries[word] = (idx_old, w_new)

    def sample(self):
        """
        Sample up to replay_ratio * vocab_size indices
        with replacement according to the current weights.
        """
        k = int(self.replay_ratio * self.vocab_size)
        if k == 0:
            return []

        words = list(self.queue)
        weights = np.array([self.entries[w][1] for w in words], dtype=np.float32)
        if weights.max() != weights.min():  # account for equal weights
            weights -= weights.min()  # offset by min
        weights /= weights.sum()  # assume already ≥ baseline

        chosen = self.rng.choices(words, weights=weights, k=k)
        return [self.entries[w][0] for w in chosen]    

    def show(self, show_weights=False):
        """
        Debug view of the buffer.
        """
        # Sort words by weight descending
        sorted_words = sorted(
            self.queue,
            key=lambda w: self.entries[w][1],
            reverse=True
        )

        if not show_weights:
            return [str(w) for w in sorted_words]

        return [f"{w}: {self.entries[w][1]:.3f}" for w in sorted_words]
    
    def stats(self, word: str):
        """
        Return (orig_idx, rank, weight) for `word`.
        rank = position in weight‑sorted buffer (1 = highest weight).
        If the word is not yet in the buffer, weight = init_w and rank = None.
        """
        if word not in self.entries:
            idx = self.vocab.index(word)
            return idx, None, self.init_w

        idx, weight = self.entries[word]
        # 1‑based rank among current buffer contents
        rank = 1 + sum(w > weight for _, w in self.entries.values())
        return idx, rank, weight
