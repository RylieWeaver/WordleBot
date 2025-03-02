import copy
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from simulation_utils import collect_episodes, process_episodes, process_episodes_mcts
from train_utils import calculate_loss, evolve_learning_params
from debug_utils import examine_gradients, examine_parameters


############################################
# TRAINING LOOP
############################################
def train(
    actor_critic_net,
    vocab,
    lr,
    batch_size,
    epochs,
    max_guesses=6,
    gamma=0.99,
    lam=0.95,
    actor_coef=0.5,
    critic_coef=1.0,
    entropy_coef=0.03,
    kl_coef=0.03,
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
    train_loader = DataLoader(vocab, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(vocab, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(actor_critic_net.parameters(), lr=lr)

    min_lr = lr / 1e1
    lr_decay_factor = 0.099
    patience = 10
    alpha = 1.0
    min_alpha = 0.03
    temperature = 3.0
    min_temperature = 0.01

    best_test_actor_loss = float("inf")
    best_test_critic_loss = float("inf")
    no_improve_count = 0

    for epoch in range(epochs):
        # ------------------- INSTANTIATE POLICIES -------------------
        old_policy_net = copy.deepcopy(actor_critic_net).eval()  # freeze old policy
        actor_critic_net.train()

        for batch_idx, target_words in enumerate(train_loader):
            # -------- Run episodes using the old policy --------
            (
                states_batch,
                old_probs_batch,
                rewards_batch,
                guess_mask_batch,
                guess_words_batch,
                correct_mask_batch,
                active_mask_batch,
            ) = collect_episodes(
                old_policy_net,
                vocab,
                target_words,
                alpha,
                temperature,
                max_guesses,
                argmax=False,
            )

            # ---------------- Process Episodes ----------------
            advantages_batch, probs_batch = process_episodes_mcts(
                actor_critic_net,
                states_batch,
                rewards_batch,
                correct_mask_batch,
                active_mask_batch,
                alpha,
                temperature,
                gamma,
                lam,
            )

            # -------------- Compute Loss --------------
            loss, actor_loss, critic_loss, entropy_loss, kl_loss = calculate_loss(
                advantages_batch,
                old_probs_batch,
                probs_batch,
                actor_coef,
                critic_coef,
                entropy_coef,
                kl_coef,
                guess_mask_batch,
                active_mask_batch,
                norm=True,
            )

            # -------------- Backprop --------------
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic_net.parameters(), max_norm=10.0)
            optimizer.step()

        # ---------------- Evaluate Learning on Full Vocab ----------------
        (
            test_loss,
            test_actor_loss,
            test_critic_loss,
            test_entropy_loss,
            test_kl_loss,
            test_accuracy,
        ) = test(
            actor_critic_net,
            test_loader,
            vocab,
            alpha,
            temperature,
            max_guesses,
            gamma,
            lam,
            actor_coef,
            0.0,  # No critic loss in evaluation
            0.0,  # No entropy loss in evaluation
            0.0,  # No KL loss in evaluation
        )

        # ---------------- Print Training Progress ----------------
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {test_loss:.4f} | "
            f"Acc: {test_accuracy:.2%} | "
            f"alpha={alpha:.2f}, temp={temperature:.2f}"
        )

        # ---------------- Evolve Learning ----------------
        # Check improvement on test loss
        if (test_actor_loss < best_test_actor_loss) or (
            test_critic_loss < best_test_critic_loss
        ):
            no_improve_count = 0
        else:
            no_improve_count += 1
        best_test_actor_loss = min(test_actor_loss, best_test_actor_loss)
        best_test_critic_loss = min(test_critic_loss, best_test_critic_loss)

        # If no improvement on test loss for 'patience' epochs => decay LR / evolve policy params alpha, temperature
        if no_improve_count >= patience:
            alpha, temperature, best_test_actor_loss, best_test_critic_loss = (
                evolve_learning_params(
                    optimizer,
                    alpha,
                    min_alpha,
                    temperature,
                    min_temperature,
                    lr,
                    min_lr,
                    lr_decay_factor,
                    best_test_actor_loss,
                    best_test_critic_loss,
                )
            )
            no_improve_count = 0

    print("Training complete!")


def test(
    actor_critic_net,
    test_loader,
    vocab,
    alpha,
    temperature,
    max_guesses=6,
    gamma=0.99,
    lam=0.95,
    actor_coef=0.5,
    critic_coef=0.0,
    entropy_coef=0.0,
    kl_coef=0.0,
):
    """
    Evaluate the model on the entire test_loader dataset.
    """
    actor_critic_net.eval()
    test_loss = 0.0
    test_actor_loss = 0.0
    test_critic_loss = 0.0
    test_entropy_loss = 0.0
    test_kl_loss = 0.0
    test_correct = 0
    test_samples = 0
    # Dummy exploration variables because we have argmax=True
    alpha = 0.0
    temperature = 1.00

    with torch.no_grad():
        for batch_idx, target_words in enumerate(test_loader):
            # -------- Run episodes --------
            (
                states_batch,
                probs_batch,
                rewards_batch,
                guess_mask_batch,
                guess_words_batch,
                active_mask_batch,
            ) = collect_episodes(
                actor_critic_net,
                vocab,
                target_words,
                alpha,
                temperature,
                max_guesses,
                argmax=True,
            )

            # ---------------- Process Episodes ----------------
            advantages_batch, probs_batch = process_episodes(
                actor_critic_net,
                states_batch,
                rewards_batch,
                active_mask_batch,
                alpha,
                temperature,
                gamma,
                lam,
            )

            # -------------- Compute Loss --------------
            (
                batch_loss,
                batch_actor_loss,
                batch_critic_loss,
                batch_entropy_loss,
                batch_kl_loss,
            ) = calculate_loss(
                advantages_batch,
                probs_batch,
                probs_batch,
                actor_coef,
                critic_coef,
                entropy_coef,
                kl_coef,
                guess_mask_batch,
                active_mask_batch,
                norm=True,
            )
            test_loss += batch_loss.item()
            test_actor_loss += batch_actor_loss.item()
            test_critic_loss += batch_critic_loss.item()
            test_entropy_loss += batch_entropy_loss.item()
            test_kl_loss += batch_kl_loss.item()
            test_correct += (
                ((active_mask_batch[:, :-1] == 0).any(dim=1)).sum().item()
            )  # the last column should be zeros anyways
            test_samples += len(target_words)

    # Calculate metrics
    avg_test_loss = test_loss / len(test_loader)
    avg_test_actor_loss = test_actor_loss / len(test_loader)
    avg_test_critic_loss = test_critic_loss / len(test_loader)
    avg_test_entropy_loss = test_entropy_loss / len(test_loader)
    avg_test_kl_loss = test_kl_loss / len(test_loader)
    test_accuracy = test_correct / test_samples

    return (
        avg_test_loss,
        avg_test_actor_loss,
        avg_test_critic_loss,
        avg_test_entropy_loss,
        avg_test_kl_loss,
        test_accuracy,
    )
