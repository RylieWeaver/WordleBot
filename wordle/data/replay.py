# General
import json

# Torch
import torch



class HardWordBuffer:
    """
    Hard-word buffer. Stores the whole target vocabulary with
    EMA difficulty weights that converge to the miss-rate.
    Sampling draws with replacement according to weights.
    """

    def __init__(self, vocab, replay_ratio=0.1, rho=0.05):
        self.vocab = vocab
        self.replay_ratio = replay_ratio
        self.init_w = 1.0 / len(vocab)
        self.weights = torch.ones(len(vocab), dtype=torch.float32) * self.init_w
        self.rho = rho

    @staticmethod
    def _flatten(t):
        return t.reshape(-1) if isinstance(t, torch.Tensor) else torch.as_tensor(t)

    @torch.no_grad()
    def update(self, selected_idx, missed_idx):
        selected = self._flatten(selected_idx)
        missed = self._flatten(missed_idx)

        # Count tries and misses
        tries_cnt = torch.bincount(selected, minlength=self.weights.numel())
        miss_cnt = torch.bincount(missed, minlength=self.weights.numel())

        # Update weights with EMA
        miss_rate = (miss_cnt.float() / tries_cnt.float().clamp(min=1)).cpu()
        self.weights = self.rho * miss_rate + (1 - self.rho) * self.weights

    def sample(self):
        k = int(self.replay_ratio * len(self.vocab))
        if k == 0:
            return []

        # Normalize weights to remove baseline and sum to 1
        if self.weights.min() != self.weights.max():
            probs = self.weights - self.weights.min()
        else:
            probs = self.weights
        probs = probs / probs.sum()

        chosen = torch.multinomial(probs, k, replacement=True)
        return chosen.tolist()

    def show(self, top_k=10):
        """
        Debug view of the buffer.
        """
        sorted_idx = torch.argsort(self.weights, descending=True)[:top_k]
        return {self.vocab[i]: round(self.weights[i].item(), 5) for i in sorted_idx}
    
    def save(self, path):
        """
        Save the buffer state to a file.
        """
        state = {
            'vocab': self.vocab,
            'weights': self.weights.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(state, f)
    
    @classmethod
    def load(cls, path, **kwargs):
        """
        Load the buffer state from a file.
        """
        with open(path, 'r') as f:
            buffer_state = json.load(f)
        buffer = cls(buffer_state['vocab'], **kwargs)
        buffer.weights = torch.tensor(buffer_state['weights'])
        return buffer
