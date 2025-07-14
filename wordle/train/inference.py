# General
import time
import traceback

# Torch
import torch

# Wordle
from wordle.environment import collect_episodes



############################################
# INFERENCE
############################################
def inference(
    actor_critic_net,
    total_vocab,
    target_vocab,
    max_guesses,
    target_idx,
    minibatch_size,
    total_vocab_tensor,
    target_vocab_tensor,
    total_vocab_states,
    target_vocab_states,
    k,
    r,
    test_search,
    m,
    alpha,
    temperature,
):
    """
    Evaluate the model on the entire test_loader dataset.
    """
    device = actor_critic_net.device
    guess_mask_batch = []
    # Dummy exploration variables because we have argmax=True
    alpha = 0.0
    temperature = 1.00

    actor_critic_net.eval()

    with torch.no_grad():
        # ------------------ Wrap batches in a try-except to handle lightning strikes ------------------
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                num_minibatches = (len(target_idx) + minibatch_size - 1) // minibatch_size
                for mb in range(num_minibatches):
                    mb_idx = target_idx[mb * minibatch_size : (mb + 1) * minibatch_size]
                    target_tensor = target_vocab_tensor[mb_idx]
                    # -------- Run episodes --------
                    (alphabet_states_minibatch, guess_states_minibatch, _, expected_values_minibatch, expected_rewards_minibatch, rewards_minibatch, guess_mask_minibatch, active_mask_minibatch, valid_mask_minibatch) = collect_episodes(
                        actor_critic_net,
                        total_vocab,
                        target_vocab,
                        total_vocab_tensor,
                        target_vocab_tensor,
                        total_vocab_states,
                        target_vocab_states,
                        target_idx,
                        target_tensor,
                        alpha,
                        temperature,
                        max_guesses,
                        m=m,
                        r=r,
                        k=k,
                        search=test_search,
                        peek=0.0,
                        argmax=True,
                    )

                    # ---------------- Add to Batch ----------------
                    guess_mask_batch.append(guess_mask_minibatch)

                break
            # ---------------- Print exception errors and continue ----------------
            except RuntimeError as e:
                if attempt < max_attempts:
                    print(f"Retrying inference due to error:\n")
                    traceback.print_exc()
                else:
                    print(f"Failed to execute inference after {max_attempts} attempts due to error:\n.")
                    traceback.print_exc()
                    continue   # out of retries, continue
                time.sleep(3)   # wait 3 seconds before next try

    # ---------------- Concatenate guess mask results ----------------
    guess_mask_batch = torch.cat(guess_mask_batch, dim=0)  # [batch_size, max_guesses, total_vocab_size]
    guess_idx_batch = torch.argmax(guess_mask_batch.float(), dim=-1)[0]  # [max_guesses]

    return guess_idx_batch
