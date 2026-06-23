# General
import numpy as np

# Torch
import torch
from torch.utils.data import Dataset, DataLoader

# Wordle
from wordle.data import words_to_tensor, construct_vocab_states, get_vocab
from wordle.utils import Config, expand_var



class WordleLoaderConfig(Config):
    def __init__(
            self,
            target_vocab: list[str] = None,
            nontarget_vocab: list[str] = None,
            batch_size: int = 32,
            repeats: int = 4,
            shuffle: bool = True,
            num_workers: int = 0,
            targets_type: str = "original",
            **kwargs
    ):
        if target_vocab is None:
            target_vocab = get_vocab(vocab_type="target", targets_type=targets_type)
        if nontarget_vocab is None:
            nontarget_vocab = get_vocab(vocab_type="nontarget")
        self.target_vocab = target_vocab
        self.nontarget_vocab = nontarget_vocab
        self.batch_size = batch_size
        self.repeats = repeats
        self.shuffle = shuffle
        self.num_workers = num_workers
        super().__init__()


class WordleLoader(DataLoader):
    def __init__(self, cfg: WordleLoaderConfig):
        # Read
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.repeats = self.cfg.repeats
        self.R = self.cfg.repeats
        # Designate vocab
        # NOTE: the target vocab is at the start of the total vocab. 
        # This is important for indexing in multiple future steps!
        target_vocab = self.cfg.target_vocab
        nontarget_vocab = self.cfg.nontarget_vocab
        total_vocab = np.concatenate((target_vocab, nontarget_vocab), axis=0)
        self.target_vocab_size = self.T = len(target_vocab)
        self.total_vocab_size = self.V = len(total_vocab)
        # Instantiate dataset
        target_vocab_tensor = words_to_tensor(target_vocab)                 # [T, 5]
        total_vocab_tensor = words_to_tensor(total_vocab)                   # [V, 5]
        self.vocab_dataset = {
            "target": {
                "size": self.T,
                "vocab": target_vocab,
                "tensor": target_vocab_tensor,                              # [T, 5]
                "states": construct_vocab_states(target_vocab_tensor),      # [T, 26, 11]
            },
            "total": {
                "size": self.V,
                "vocab": total_vocab,
                "tensor": total_vocab_tensor,                               # [V, 5]
                "states": construct_vocab_states(total_vocab_tensor),       # [V, 26, 11]
            }
        }
        # Simply shuffle idxs
        idx_dataset = torch.arange(len(target_vocab))
        super().__init__(
            idx_dataset,
            batch_size=self.batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate,
        )

    def _repeat_idx(self, idx):                                 # [B, *] (* can be empty)
        if self.R == 1:
            return idx.unsqueeze(-1)                            # [B, *, 1]
        return expand_var(idx, dim=1, size=self.R)              # [B, R] -> [B, R, *]
    
    def _idx2data(self, idx):                                   # [*]
        data = self.vocab_dataset
        return {
            "target": {**data["target"], "batch_idx": idx},      # [*]
            "total": {**data["total"], "batch_idx": idx},        # [*]
        }

    def _collate(self, batch):
        idx = torch.as_tensor(batch, dtype=torch.long)          # [B]
        idx = self._repeat_idx(idx)                             # [B, R]
        batch = self._idx2data(idx)
        return batch
    

class EpisodesDataset(Dataset):
    def __init__(self, episodes: dict[str, dict[str, torch.Tensor]]):
        self.episodes = episodes
        self.n = episodes["states"]["alphabet"].shape[0]

    def __len__(self):
        return self.n

    def _slice_dict(self, dict, i):
        return {k: v[i] for k, v in dict.items()}
    
    def slice_episodes(self, i):
        episodes = {
            "states": self._slice_dict(self.episodes["states"], i),
            "actions": self._slice_dict(self.episodes["actions"], i),
            "responses": self._slice_dict(self.episodes["responses"], i),
        }
        return episodes

    def __getitem__(self, i):
        return self.slice_episodes(i)
