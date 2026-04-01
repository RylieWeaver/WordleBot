# # General
# import time
# import traceback

# # Torch
# import torch

# # Wordle



# ############################################
# # INFERENCE
# ############################################
# def inference(
#     actor_critic_net,
#     total_vocab,
#     target_vocab,
#     max_guesses,
#     target_idx,
#     minibatch_size,
#     total_vocab_tensor,
#     target_vocab_tensor,
#     total_vocab_states,
#     target_vocab_states,
#     m,
#     alpha,
#     temperature,
# ):
#     """
#     Evaluate the model on the entire test_loader dataset.
#     """
#     device = actor_critic_net.device
#     guess_mask_batch = []
#     active_mask_batch = []
#     # Dummy exploration variables because we have argmax=True
#     alpha = 0.0
#     temperature = 1.00

#     actor_critic_net.eval()

#     with torch.no_grad():
#         # ------------------ Wrap batches in a try-except to handle lightning strikes ------------------
#         max_attempts = 3
#         for attempt in range(1, max_attempts + 1):
#             try:
#                 num_minibatches = (len(target_idx) + minibatch_size - 1) // minibatch_size
#                 for mb in range(num_minibatches):
#                     mb_idx = target_idx[mb * minibatch_size : (mb + 1) * minibatch_size]
#                     target_tensor = target_vocab_tensor[mb_idx]
#                     # -------- Run episodes --------
#                     (alphabet_states_minibatch, guess_states_minibatch, _, correct_minibatch, expected_values_minibatch, expected_rewards_minibatch, rewards_minibatch, guess_mask_minibatch, active_mask_minibatch, valid_mask_minibatch) = collect_episodes(
#                         actor_critic_net,
#                         total_vocab,
#                         target_vocab,
#                         total_vocab_tensor,
#                         target_vocab_tensor,
#                         total_vocab_states,
#                         target_vocab_states,
#                         target_idx,
#                         target_tensor,
#                         alpha,
#                         temperature,
#                         max_guesses,
#                         m=m,
#                         argmax=True,
#                     )

#                     # ---------------- Add to Batch ----------------
#                     guess_mask_batch.append(guess_mask_minibatch)  # [batch_size, max_guesses, total_vocab_size]
#                     active_mask_batch.append(active_mask_minibatch)  # [batch_size, max_guesses+1]

#                 break
#             # ---------------- Print exception errors and continue ----------------
#             except RuntimeError as e:
#                 if attempt < max_attempts:
#                     print(f"Retrying inference due to error:\n")
#                     traceback.print_exc()
#                 else:
#                     print(f"Failed to execute inference after {max_attempts} attempts due to error:\n.")
#                     traceback.print_exc()
#                     continue   # out of retries, continue
#                 time.sleep(3)   # wait 3 seconds before next try

#     # ---------------- Concatenate guess mask results ----------------
#     guess_mask_batch = torch.cat(guess_mask_batch, dim=0)  # [batch_size, max_guesses, total_vocab_size]
#     guess_idx_batch = torch.argmax(guess_mask_batch.float(), dim=-1)[0]  # [max_guesses]
#     active_mask_batch = torch.cat(active_mask_batch, dim=0)  # [batch_size, max_guesses+1]

#     return guess_idx_batch, active_mask_batch





# General
import json
import argparse
from pathlib import Path
import numpy as np

# Torch
import torch

# Wordle
from wordle.data import get_vocab, words_to_tensor, WordleLoaderConfig
from wordle.environment import SimulatorConfig, Simulator
from wordle.model import name2model
from wordle.utils import resolve_device



def load_model_ckpt(model_cfg_path: Path, model_state_path: Path, device=None):
    # Load model class dynamically from name2model
    with model_cfg_path.open("r") as f:
        model_cfg = json.load(f)
    # NOTE: We pop out model name so that it's not passed, even though **kwargs should catch it
    model_cfg_cls, model_cls = name2model[model_cfg.pop("model_name")]
    model_cfg = model_cfg_cls(**model_cfg)
    model = model_cls(model_cfg, device=device).to(device)
    state_dict = torch.load(model_state_path, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def inference(model, target_idx):
    # Data
    target_vocab = get_vocab(vocab_type="target")               # [T]
    nontarget_vocab = get_vocab(vocab_type="nontarget")         # [V-T]
    target_idx = torch.tensor(target_idx).squeeze()             # [T_sub]
    
    # Prepare simulator
    loader_cfg = WordleLoaderConfig(
        target_vocab=target_vocab,
        nontarget_vocab=nontarget_vocab,
        batch_size=32,
        repeats=1,
        shuffle=False,
    )
    simulator_cfg = SimulatorConfig(
        loader_cfg=loader_cfg,
        gamma=0.2,
        lam=0.95,
        max_guesses=6,
        m=1,
    )
    simulator = Simulator(simulator_cfg)

    # Collect with eval mode and argmax policy
    model.eval()
    data = simulator.loader._collate(target_idx)
    episodes = simulator.collect_episodes_mb(
        model,
        data,
        alpha=0.0,
        temperature=1.0,
        argmax=True,
        desc="Collecting Eval Episodes"
    )
    return episodes



if __name__ == '__main__':
    # Read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help="Path to model dir.")
    args = parser.parse_args()
    model_cfg_path = Path(args.model_cfg_path) / "config.json"
    model_state_path = Path(args.model_cfg_path) / "model.pt"

    # Setup
    device = resolve_device("mps")

    # Model
    model = load_model_ckpt(model_cfg_path=model_cfg_path, model_state_path=model_state_path, device=device)

    # Run inference collection
    episodes = inference(model)
    print("DONE!")
