# General
import json
import random

# Torch
import torch



class HardWordBuffer:
    """
    Hard-word buffer. Stores the whole target vocabulary with
    EMA difficulty weights that converge to the miss-rate.
    Sampling draws with replacement according to weights.
    """

    def __init__(self, target_vocab, replay_ratio=0.1):
        self.vocab = target_vocab
        self.replay_ratio = replay_ratio
        self.tries = torch.ones(len(target_vocab), dtype=torch.int32)  # Number of tries for each word
        self.misses = torch.ones(len(target_vocab), dtype=torch.int32)  # Number of misses for each word

    @staticmethod
    def _flatten(t):
        return t.view(-1) if isinstance(t, torch.Tensor) else torch.as_tensor(t)

    @torch.no_grad()
    def update(self, selected_idx, missed_idx):
        selected = self._flatten(selected_idx)
        missed = self._flatten(missed_idx)

        # Count tries and misses
        tries_cnt = torch.bincount(selected, minlength=self.weights.numel())
        miss_cnt = torch.bincount(missed, minlength=self.weights.numel())
        self.tries += tries_cnt
        self.misses += miss_cnt

    def sample(self, batch_size):
        k = int(self.replay_ratio * batch_size)
        if k == 0:
            return []

        probs = self.weights
        probs = probs / probs.sum()

        chosen = torch.multinomial(probs, k, replacement=True)
        return chosen.tolist()

    def show(self, show_weights=False, top_k=10):
        """
        Debug view of the buffer.
        """
        sorted_idx = torch.argsort(self.weights, descending=True)[:top_k]
        return {self.vocab[i]: self.weights[i].item() for i in sorted_idx}
    
    def save(self, path):
        """
        Save the buffer state to a file.
        """
        state = {
            'vocab': self.vocab,
            'weights': self.weights.tolist(),
            'count': self.count.tolist(),
            'tries': self.tries.tolist(),
            'replay_ratio': self.replay_ratio,
            'rho': self.rho,
            'init_w': self.init_w
        }
        with open(path, 'w') as f:
            json.dump(state, f)
    
    @classmethod
    def load(cls, path):
        """
        Load the buffer state from a file.
        """
        with open(path, 'r') as f:
            state = json.load(f)
        buffer = cls(state['vocab'], state['replay_ratio'], state['rho'], state['init_w'])
        buffer.weights = torch.tensor(state['weights'])
        buffer.count = torch.tensor(state['count'])
        buffer.tries = torch.tensor(state['tries'])
        return buffer
