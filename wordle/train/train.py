# General
import os
import copy
import time
from collections import deque


# Torch
import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.optim as optim

# Wordle
from wordle.data import words_to_tensor, tensor_to_words, construct_vocab_states, HardWordBuffer
from wordle.environment import collect_episodes, process_episodes
from wordle.train import calculate_loss, evolve_learning_params
from wordle.utils import measure_grad_norms, save_checkpoint, rest_computer



############################################
# TRAINING LOOP
############################################
def train(
    actor_critic_net,
    total_vocab,
    target_vocab,
    max_guesses,
    lr,
    batch_size,
    minibatch_size,
    epochs,
    k,
    r,
    train_search,
    test_search,
    gamma,
    lam,
    m,
    actor_coef,
    critic_coef,
    entropy_coef,
    kl_reg_coef,
    kl_guide_coef,
    kl_best_coef,
    peek,
    alpha,
    min_alpha,
    alpha_step,
    temperature,
    min_temperature,
    temperature_decay_factor,
    checkpointing,
    log_dir,
    min_lr_factor,
    global_lr_decay_factor,
    lr_decay_factor,
    greedify_patience,
    early_stopping_patience,
    config,
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
    device = actor_critic_net.device
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    total_vocab_tensor = words_to_tensor(total_vocab).to(device)  # [total_vocab_size, 5]
    target_vocab_tensor = words_to_tensor(target_vocab).to(device)  # [target_vocab_size, 5]
    total_vocab_states = construct_vocab_states(words_to_tensor(total_vocab).to(device))  # [total_vocab_size, 26, 11]
    target_vocab_states = construct_vocab_states(words_to_tensor(target_vocab).to(device))  # [target_vocab_size, 26, 11]

    train_loader = DataLoader(torch.arange(len(target_vocab)), batch_size=batch_size, shuffle=True)
    replay_loader = HardWordBuffer(target_vocab, batch_size=batch_size, capacity=max(int(0.05*len(target_vocab)), 1), replay_ratio=0.2, rho=0.1)
    test_loader = DataLoader(torch.arange(len(target_vocab)), batch_size=batch_size, shuffle=False)

    # NOTE: Testing
    # train_loader = DataLoader(
    #     torch.arange(len(target_vocab)),
    #     batch_size=batch_size,
    #     sampler=RandomSampler(
    #         torch.arange(len(target_vocab)),
    #         replacement=True,
    #         num_samples=batch_size),
    #     drop_last=True)
    # replay_loader = HardWordBuffer(target_vocab, batch_size=len(target_vocab), capacity=max(int(0.05*len(target_vocab)), 1), replay_ratio=0.2, rho=0.1)
    # test_loader = DataLoader(torch.arange(len(target_vocab)), batch_size=len(target_vocab), shuffle=False)

    replay=True
    optimizer = optim.AdamW(actor_critic_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)

    min_lr = lr * min_lr_factor

    best_test_actor_loss = float('inf')
    best_test_critic_loss = float('inf')
    best_accuracy = 0.0
    best_guesses = float(max_guesses)
    no_improve_count = 0

    best_policy_net = copy.deepcopy(actor_critic_net).eval().to(device)
    window_size = 3
    # policy_buffer = deque(maxlen=window_size)
    # policy_buffer.append(copy.deepcopy(actor_critic_net).cpu())
    for epoch in range(epochs):
        # ------------------- INSTANTIATE NETWORKS -------------------
        actor_critic_net.train()
        if epoch%window_size == 0:
            old_policy_net = copy.deepcopy(actor_critic_net)

        for batch_idx, target_idx in enumerate(train_loader):
            # old_policy_net = policy_buffer[0].to(device).eval()  # Get the policy from window_size-1 steps ago for KL-reg
            if replay:
                replay_idx = torch.tensor(replay_loader.sample())
                selected_idx = torch.cat((target_idx, replay_idx), dim=0)  # [batch_size + replay_ratio * batch_size]
            else:
                selected_idx = target_idx
            mb_size = minibatch_size if minibatch_size is not None else len(selected_idx)
            # ------------------ Wrap batches in a try-except to handle lightning strikes ------------------
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    correct_batch = []
                    for start in range(0, len(selected_idx), mb_size):
                        mb_idx = selected_idx[start:start+mb_size]
                        mb_proportion = (len(mb_idx) / len(selected_idx))
                        target_tensor = target_vocab_tensor[mb_idx]
                        # -------- Collect episodes using the old policy --------
                        (alphabet_states_minibatch, guess_states_minibatch, old_probs_minibatch, guide_probs_minibatch, expected_values_minibatch, expected_rewards_minibatch, rewards_minibatch, guess_mask_minibatch, active_mask_minibatch, valid_mask_minibatch) = collect_episodes(
                            # old_policy_net,
                            actor_critic_net,
                            total_vocab,
                            target_vocab,
                            total_vocab_tensor,
                            target_vocab_tensor,
                            total_vocab_states,
                            target_vocab_states,
                            mb_idx,
                            target_tensor,
                            alpha,
                            temperature,
                            max_guesses,
                            k=k,
                            r=r,
                            m=m,
                            search=train_search,
                            peek=peek,
                            argmax=False,
                        )

                        # ---------------- Process Episodes Best Checkpoint ----------------
                        _, best_probs_minibatch, _ = process_episodes(
                            best_policy_net,
                            alphabet_states_minibatch,
                            guess_states_minibatch,
                            expected_values_minibatch,
                            expected_rewards_minibatch,
                            rewards_minibatch,
                            active_mask_minibatch,
                            valid_mask_minibatch,
                            alpha,
                            temperature,
                            gamma,
                            lam,
                        )

                        # ---------------- Process Episodes Actual Net ----------------
                        advantages_minibatch, probs_minibatch, correct_minibatch = process_episodes(
                            actor_critic_net,
                            alphabet_states_minibatch,
                            guess_states_minibatch,
                            expected_values_minibatch,
                            expected_rewards_minibatch,
                            rewards_minibatch,
                            active_mask_minibatch,
                            valid_mask_minibatch,
                            alpha,
                            temperature,
                            gamma,
                            lam,
                        )

                        # -------------- Compute Loss --------------
                        # The KL divergence should be 0 on the first batch in an epoch, but it is often not 
                        # because of a non-deterministic forward (e.g. dropout). Setting to 0 removes some noise
                        (loss_mb, actor_loss_mb, critic_loss_mb, entropy_loss_mb, kl_reg_loss_mb, kl_guide_loss_mb, kl_best_loss_mb) = calculate_loss(
                            advantages_minibatch,
                            old_probs_minibatch,
                            guide_probs_minibatch,
                            best_probs_minibatch,
                            probs_minibatch,
                            actor_coef,
                            critic_coef,
                            entropy_coef,
                            kl_reg_coef,
                            kl_guide_coef,
                            kl_best_coef,
                            guess_mask_minibatch,
                            active_mask_minibatch,
                            valid_mask_minibatch,
                            norm=True,
                        )

                        # ------------------ Normalize Minibatch Losses ------------------
                        # Total loss should be invariant to minibatch size
                        loss_mb = loss_mb / mb_proportion
                        actor_loss_mb = actor_loss_mb / mb_proportion
                        critic_loss_mb = critic_loss_mb / mb_proportion
                        entropy_loss_mb = entropy_loss_mb / mb_proportion
                        kl_reg_loss_mb = kl_reg_loss_mb / mb_proportion
                        kl_guide_loss_mb = kl_guide_loss_mb / mb_proportion
                        kl_best_loss_mb = kl_best_loss_mb / mb_proportion

                        # ---------------- Measure Grad Norms ----------------
                        if (epoch%25 == 0) and (batch_idx == len(train_loader)-1) and (mb_idx[-1] == selected_idx[-1]):  # Measure grad norms on the last minibatch of the last batch of every epochs
                            actor_grad_norm, critic_grad_norm, entropy_grad_norm, kl_reg_grad_norm, kl_guide_grad_norm, kl_best_grad_norm = measure_grad_norms(
                                actor_critic_net,
                                actor_loss_mb, actor_coef,
                                critic_loss_mb, critic_coef,
                                entropy_loss_mb, entropy_coef,
                                kl_reg_loss_mb, kl_reg_coef,
                                kl_guide_loss_mb, kl_guide_coef,
                                kl_best_loss_mb, kl_best_coef,
                            )
                            print(f"Actor grad norm: {actor_grad_norm:.4f}, Critic grad norm: {critic_grad_norm:.4f}, Entropy grad norm: {entropy_grad_norm:.4f}, KL-Reg grad norm: {kl_reg_grad_norm:.4f}, KL-Guide grad norm: {kl_guide_grad_norm:.4f}, KL-Best grad norm: {kl_best_grad_norm:.4f}")

                        # ------------------ Accumulate Minibatch Results ------------------
                        loss_mb.backward()
                        correct_batch.append(correct_minibatch)

                    # ---------------- Concatenate correct for entire batch ----------------
                    correct_batch = torch.cat(correct_batch, dim=0)

                    # ---------------- Update Buffer w/ Policy ----------------
                    # policy_buffer.append(copy.deepcopy(actor_critic_net).cpu())

                    # -------------- Backprop --------------
                    torch.nn.utils.clip_grad_norm_(actor_critic_net.parameters(), max_norm=3.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    # ---------------- Update Replay ----------------
                    replay_loader.update(selected_idx, selected_idx[~correct_batch.cpu()])

                    break
                # ---------------- Print exception errors and continue ----------------
                except RuntimeError as e:
                    print(f"Skipping batch {batch_idx} in epoch {epoch} due to error:\n{e}")
                    if attempt == max_attempts:
                        continue   # out of retries, raise error
                    time.sleep(3)   # wait 3 seconds before next try

        # ---------------- Evaluate Learning on Full Vocab ----------------
        (test_loss, test_actor_loss, test_critic_loss, test_entropy_loss, test_kl_reg_loss, test_kl_guide_loss, test_kl_best_loss, test_correct, test_accuracy, test_guesses,) = test(
            best_policy_net,
            actor_critic_net,
            total_vocab,
            target_vocab,
            max_guesses,
            test_loader,
            minibatch_size if minibatch_size is not None else batch_size,
            total_vocab_tensor,
            target_vocab_tensor,
            total_vocab_states,
            target_vocab_states,
            k,
            r,
            test_search,
            gamma,
            lam,
            m,
            actor_coef,
            0.0,  # No critic loss in evaluation
            0.0,  # No entropy loss in evaluation
            0.0,  # No KL reg loss in evaluation
            0.0,  # No KL guide loss in evaluation
            0.0,  # No KL best loss in evaluation
            alpha,
            temperature,
        )

        # ---------------- Print Training Progress and Write to Run Log ----------------
        log_line = (
            f"Epoch {epoch}/{epochs} | Acc: {test_accuracy:.2%} | Avg Guesses: {test_guesses:.2f} | "
            f"alpha={alpha:.2f}, temp={temperature:.2f} | Loss: {test_loss:.4f} | "
            f"Actor Loss: {test_actor_loss:.4f} * {actor_coef:.2f}, Critic Loss: {test_critic_loss:.4f} * {critic_coef:.2f}| "
            f"Entropy Loss: {test_entropy_loss:.4f} * {entropy_coef:.2f}, Train KL-Reg Loss: {kl_reg_loss_mb.item():.4f} * {kl_reg_coef:.2f}, "
            f"Test KL-Guide Loss: {test_kl_guide_loss:.4f} * {kl_guide_coef:.2f}, Test KL-Best Loss: {test_kl_best_loss:.4f} * {kl_best_coef:.2f}, "
        )
        print(log_line)
        with open(f'{log_dir}/run_log.txt', 'a') as f:
            f.write(log_line + "\n")

        # ---------------- Checkpoint ----------------
        replay = (test_accuracy < 1.0)  # use replay if any guesses were incorrect
        if (test_accuracy > best_accuracy or (test_accuracy == best_accuracy and test_guesses < best_guesses)):
            print(f'  -> New best model found')
            best_policy_net = copy.deepcopy(actor_critic_net).eval().to(device)
            if checkpointing:
                save_checkpoint(actor_critic_net, test_accuracy, test_guesses, config, log_dir)

        # ---------------- Evolve Learning ----------------
        # Check improvement on test loss
        if ((test_actor_loss < best_test_actor_loss) or (test_critic_loss < best_test_critic_loss) or (test_accuracy > best_accuracy) or (test_accuracy == best_accuracy and test_guesses < best_guesses)):
            no_improve_count = 0
        else:
            no_improve_count += 1
        # If no improvement on test loss for 'greedify_patience' epochs => decay LR / evolve policy params alpha, temperature
        if no_improve_count >= greedify_patience:
            (lr, min_lr, alpha, temperature, best_test_actor_loss, best_test_critic_loss,) = evolve_learning_params(
                optimizer,
                alpha,
                min_alpha,
                alpha_step,
                temperature,
                min_temperature,
                temperature_decay_factor,
                lr,
                min_lr,
                global_lr_decay_factor,
                lr_decay_factor,
                best_test_actor_loss,
                best_test_critic_loss,
            )
            no_improve_count = 0
            # kl_reg_coef = kl_reg_coef * 0.99
            # kl_guide_coef = kl_guide_coef * 0.99
            # kl_best_coef = kl_best_coef * 0.99

        # ----------------- Update Statistics ----------------
        best_accuracy = max(test_accuracy, best_accuracy)
        best_guesses = min(test_guesses, best_guesses)
        best_test_actor_loss = min(test_actor_loss, best_test_actor_loss)
        best_test_critic_loss = min(test_critic_loss, best_test_critic_loss)

        # ---------------- Overall Early Stopping ----------------
        if no_improve_count >= early_stopping_patience:
            print(f'No improvement for {no_improve_count} epochs, stopping training.')
            break

        # ---------------- Rest Computer to Prevent Overheating ----------------
        rest_computer(len(target_vocab))  # Rest based on target vocab length (proxy for experiment size)
            
    print('Training complete!')



############################################
# TESTING LOOP
############################################
def test(
    best_policy_net,
    actor_critic_net,
    total_vocab,
    target_vocab,
    max_guesses,
    test_loader,
    minibatch_size,
    total_vocab_tensor,
    target_vocab_tensor,
    total_vocab_states,
    target_vocab_states,
    k,
    r,
    test_search,
    gamma,
    lam,
    m,
    actor_coef,
    critic_coef,
    entropy_coef,
    kl_reg_coef,
    kl_guide_coef,
    kl_best_coef,
    alpha,
    temperature,
):
    """
    Evaluate the model on the entire test_loader dataset.
    """
    device = actor_critic_net.device
    actor_critic_net.eval()
    # test_loss = test_actor_loss = test_critic_loss = test_entropy_loss = test_kl_reg_loss = test_kl_guide_loss = test_kl_best_loss = torch.zeros(1, device=device)
    test_loss = test_actor_loss = test_critic_loss = test_entropy_loss = test_kl_reg_loss = test_kl_guide_loss = test_kl_best_loss = 0.0
    test_guesses = test_samples = 0.0
    test_correct = []
    # Dummy exploration variables because we have argmax=True
    alpha = 0.0
    temperature = 1.00

    with torch.no_grad():
        for batch_idx, target_idx in enumerate(test_loader):
            # ------------------ Wrap batches in a try-except to handle lightning strikes ------------------
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    for start in range(0, len(target_idx), minibatch_size):
                        mb_idx = target_idx[start:start+minibatch_size]
                        target_tensor = target_vocab_tensor[mb_idx]
                        # -------- Run episodes --------
                        (alphabet_states_minibatch, guess_states_minibatch, old_probs_minibatch, guide_probs_minibatch, expected_values_minibatch, expected_rewards_minibatch, rewards_minibatch, guess_mask_minibatch, active_mask_minibatch, valid_mask_minibatch) = collect_episodes(
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

                        # ---------------- Process Episodes Best Checkpoint ----------------
                        _, best_probs_minibatch, _ = process_episodes(
                            best_policy_net,
                            alphabet_states_minibatch,
                            guess_states_minibatch,
                            expected_values_minibatch,
                            expected_rewards_minibatch,
                            rewards_minibatch,
                            active_mask_minibatch,
                            valid_mask_minibatch,
                            alpha,
                            temperature,
                            gamma,
                            lam,
                        )

                        # ---------------- Process Episodes ----------------
                        advantages_minibatch, probs_minibatch, correct_minibatch = process_episodes(
                            actor_critic_net,
                            alphabet_states_minibatch,
                            guess_states_minibatch,
                            expected_values_minibatch,
                            expected_rewards_minibatch,
                            rewards_minibatch,
                            active_mask_minibatch,
                            valid_mask_minibatch,
                            alpha,
                            temperature,
                            gamma,
                            lam,
                        )

                        # -------------- Compute Loss --------------
                        (loss_mb, actor_loss_mb, critic_loss_mb, entropy_loss_mb, kl_reg_loss_mb, kl_guide_loss_mb, kl_best_loss_mb) = calculate_loss(
                            advantages_minibatch,
                            old_probs_minibatch,
                            guide_probs_minibatch,
                            best_probs_minibatch,
                            probs_minibatch,
                            actor_coef,
                            critic_coef,
                            entropy_coef,
                            kl_reg_coef,
                            kl_guide_coef,
                            kl_best_coef,
                            guess_mask_minibatch,
                            active_mask_minibatch,
                            valid_mask_minibatch,
                            norm=True,
                        )

                        # ------------------ Accumulate Minibatch Results ------------------
                        test_loss = test_loss + loss_mb.item()
                        test_actor_loss = test_actor_loss + actor_loss_mb.item()
                        test_critic_loss = test_critic_loss + critic_loss_mb.item()
                        test_entropy_loss = test_entropy_loss + entropy_loss_mb.item()
                        test_kl_reg_loss = test_kl_reg_loss + kl_reg_loss_mb.item()
                        test_kl_guide_loss = test_kl_guide_loss + kl_guide_loss_mb.item()
                        test_kl_best_loss = test_kl_best_loss + kl_best_loss_mb.item()
                        test_correct.append(correct_minibatch)
                        test_guesses += (active_mask_minibatch[:, :-1].sum(dim=-1).sum(dim=-1)).item()
                        test_samples += len(mb_idx)  # total number of samples

                    break
                # ---------------- Print exception errors and continue ----------------
                except RuntimeError as e:
                    if attempt < max_attempts:
                        print(f"Retrying batch {batch_idx} in testing due to error:\n{e}")
                    else:
                        print(f"Failed to process batch {batch_idx} after {max_attempts} attempts.")
                        continue   # out of retries, continue
                    time.sleep(3)   # wait 3 seconds before next try

    # ---------------- Concatenate minibatch results ----------------
    test_correct = torch.cat(test_correct, dim=0)

    # Calculate metrics
    batches = float(len(test_loader))
    avg_test_loss = (test_loss / batches)
    avg_test_actor_loss = (test_actor_loss / batches)
    avg_test_critic_loss = (test_critic_loss / batches)
    avg_test_entropy_loss = (test_entropy_loss / batches)
    avg_test_kl_reg_loss = (test_kl_reg_loss / batches)
    avg_test_kl_guide_loss = (test_kl_guide_loss / batches)
    avg_test_kl_best_loss = (test_kl_best_loss / batches)
    test_accuracy = (test_correct.sum().item() / test_samples)
    test_guesses = (test_guesses / test_samples)  # average guesses per sample

    return (
        avg_test_loss,
        avg_test_actor_loss,
        avg_test_critic_loss,
        avg_test_entropy_loss,
        avg_test_kl_reg_loss,
        avg_test_kl_guide_loss,
        avg_test_kl_best_loss,
        test_correct,
        test_accuracy,
        test_guesses,
    )
