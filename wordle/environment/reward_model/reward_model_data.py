# General
import copy

# Torch
import torch
from torch.utils.data import DataLoader

# Wordle
from wordle.data import get_vocab, words_to_tensor, construct_vocab_states
from wordle.environment import collect_episodes, calculate_alphabet_entropy
from wordle.model import ActorCriticNet


# ---------------------------------------------------------------
def collect_data_with_model(
    actor_critic_net,
    vocab,
    batch_size,
    epochs,
    max_guesses=6,
):
    """
    Actor-Critic train loop with single-episode simulation,
    but parameter updates happen after collecting a batch of episodes.

    Key points:
      - We store the raw advantages separately (rather than -log_prob * advantage).
      - We collect an entropy term from each step to encourage exploration.
      - We normalize those advantages and multiply by log_prob at the end.
      - The entropy term is added to the total loss with a negative sign (so that
        maximizing entropy reduces the loss).
    """
    # ---------------- Setup ----------------
    vocab_tensor = words_to_tensor(vocab).to(actor_critic_net.device)  # shape [vocab_size, 5]
    vocab_states = construct_vocab_states(vocab_tensor).to(actor_critic_net.device)  # shape [vocab_size, 26, 11]
    states = []
    entropies = []

    loader = DataLoader(vocab_tensor, batch_size=batch_size, shuffle=True)

    alpha = 1.00
    temperature = 3.0

    for epoch in range(epochs):
        # ------------------- INSTANTIATE NETWORKS -------------------
        old_policy_net = copy.deepcopy(actor_critic_net).eval()  # freeze old policy
        actor_critic_net.train()

        for batch_idx, target_tensor in enumerate(loader):
            # -------- Collect episodes --------
            (states_batch, old_probs_batch, rewards_batch, guess_mask_batch, active_mask_batch,) = collect_episodes(
                old_policy_net,
                vocab,
                vocab_tensor,
                vocab_states,
                target_tensor,
                alpha,
                temperature,
                max_guesses,
                argmax=False,
            )
            alphabet_states_batch = states_batch.view(-1, 26*11+max_guesses)[:, :-max_guesses].view(-1, 26, 11)  # shape [batch_size*max_guesses, 26*11+6] --> [batch_size*max_guesses, 26*11]
            target_mask = torch.ones([states_batch.shape[0], len(vocab)], dtype=torch.bool).to(states_batch.device)  # shape [batch_size*max_guesses]
            entropies_batch, new_target_mask = calculate_alphabet_entropy(target_mask, alphabet_states_batch, vocab_states)

            # Add to lists
            states.append(alphabet_states_batch)
            entropies.append(entropies_batch)

    # Convert to tensors
    states = torch.stack(states).view(-1, 26, 11)  # shape [N, 26, 11]
    entropies = torch.stack(entropies).view(-1)  # shape [N]

    return states, entropies


if __name__ == "__main__":
    # Generate
    savename = "wordle_data"

    # Instantiate hyperparameters
    epochs = 100
    batch_size = 2315
    vocab_size = 2315
    max_guesses = 10  # This many guesses gives an average entropy of about 3.4 in the collected states, which is about the square root between the max (11.1768) and min (0.0) entropies.
    device = 'cpu'

    # Instantiate data
    state_size = 26*11 + max_guesses
    vocab = get_vocab(vocab_size=vocab_size)

    # Instantiate the network and optimizer
    actor_critic_net = ActorCriticNet(state_size=state_size, vocab_size=vocab_size, hidden_dim=16, layers=2, dropout=0.1, device=device).to(device)

    # Collect the data
    X, y = collect_data_with_model(actor_critic_net, vocab, batch_size, epochs, max_guesses=max_guesses)

    # Save
    torch.save((X, y), f'{savename}.pt')
