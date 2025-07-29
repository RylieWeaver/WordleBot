# General
import os
import copy
import time
import traceback

# Torch
import torch
import torch.optim as optim

# Wordle
from wordle.data import words_to_tensor, tensor_to_words, construct_vocab_states, HardWordBuffer
from wordle.environment import collect_episodes, process_episodes
from wordle.train import calculate_loss, evolve_learning_params
from wordle.utils import measure_grad_norms, save_checkpoint, clear_cache, rest_computer



############################################
# PRETRAINING LOOP
############################################
def pretrain(
    actor_critic_net,
    total_vocab,
    target_vocab,
    max_guesses,
    lr,
    batch_size,
    collect_minibatch_size,
    process_minibatch_size,
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
    temperature,
    checkpointing,
    log_dir,
    warmup_steps,
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

    replay_loader = HardWordBuffer(target_vocab, batch_size=len(target_vocab), capacity=max(int(0.05*len(target_vocab)), 1), replay_ratio=(1/len(target_vocab)), rho=0.1)

    replay=True
    optimizer = optim.AdamW(actor_critic_net.parameters(), lr=lr, weight_decay=1e-4)
    warmup_factor = 1.0 / warmup_steps
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, end_factor=1.0, total_iters=warmup_steps)

    best_test_actor_loss = float('inf')
    best_test_critic_loss = float('inf')
    best_accuracy = 0.0
    best_guesses = float(max_guesses)
    no_improve_count = 0

    old_policy_net = copy.deepcopy(actor_critic_net).eval().to(device)
    best_policy_net = copy.deepcopy(actor_critic_net).eval().to(device)
    actor_critic_net.train()

    reward_blend_factor = 0.9
    value_blend_factor = 0.1

    max_attempts = 3


    for epoch in range(1, epochs + 1):
        # ------------------- Generate Epoch Indices -------------------
        target_vocab_idx = torch.arange(len(target_vocab)).to(device)
        if replay:
            replay_idx = torch.tensor(replay_loader.sample()).to(device)
            epoch_idx = torch.cat((target_vocab_idx, replay_idx), dim=0)  # [(1 + replay_ratio) * batch_size]
        else:
            epoch_idx = target_vocab_idx
        epoch_idx = epoch_idx[torch.randperm(len(epoch_idx))]  # Shuffle the indices

        # ------------------ Wrap episode collection in a try-except to handle lightning strikes ------------------
        for attempt in range(1, max_attempts + 1):
            try:
                # -------- Collect Episodes in Minibatches --------
                alphabet_states_batch, guess_states_batch, expected_values_batch, expected_rewards_batch, rewards_batch, guess_mask_batch, active_mask_batch, valid_mask_batch = [], [], [], [], [], [], [], []
                mb_size = collect_minibatch_size if collect_minibatch_size is not None else len(epoch_idx)
                num_minibatches = (len(epoch_idx) + mb_size - 1) // mb_size
                for mb in range(num_minibatches):
                    # Slice minibatch
                    mb_idx = epoch_idx[mb*mb_size:(mb+1)*mb_size]
                    target_tensor = target_vocab_tensor[mb_idx]
                    # Collect minibatch episodes
                    (alphabet_states_minibatch, guess_states_minibatch, _, expected_values_minibatch, expected_rewards_minibatch, rewards_minibatch, guess_mask_minibatch, active_mask_minibatch, valid_mask_minibatch) = collect_episodes(
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
                    # Append to batch lists
                    alphabet_states_batch.append(alphabet_states_minibatch)
                    guess_states_batch.append(guess_states_minibatch)
                    expected_values_batch.append(expected_values_minibatch)
                    expected_rewards_batch.append(expected_rewards_minibatch)
                    rewards_batch.append(rewards_minibatch)
                    guess_mask_batch.append(guess_mask_minibatch)
                    active_mask_batch.append(active_mask_minibatch)
                    valid_mask_batch.append(valid_mask_minibatch)

                # Concatenate Batch Results
                alphabet_states_batch = torch.cat(alphabet_states_batch, dim=0)  # [batch_size, max_guesses+1, 26, 11]
                guess_states_batch = torch.cat(guess_states_batch, dim=0)  # [batch_size, max_guesses+1, 26, 11]
                expected_values_batch = torch.cat(expected_values_batch, dim=0)  # [batch_size, max_guesses+1]
                expected_rewards_batch = torch.cat(expected_rewards_batch, dim=0)  # [batch_size, max_guesses+1]
                rewards_batch = torch.cat(rewards_batch, dim=0)  # [batch_size, max_guesses+1]
                guess_mask_batch = torch.cat(guess_mask_batch, dim=0)  # [batch_size, max_guesses+1, vocab_size]
                active_mask_batch = torch.cat(active_mask_batch, dim=0)  # [batch_size, max_guesses+1]
                valid_mask_batch = torch.cat(valid_mask_batch, dim=0)  # [batch_size, max_guesses+1]

                break
            # ---------------- Print exception errors and continue ----------------
            except RuntimeError as e:
                if attempt < max_attempts:
                    print(f"Retrying episode collection due to error:\n")
                    traceback.print_exc()
                else:
                    print(f"Failed to collects episodes after {max_attempts} attempts due to error:\n.")
                    traceback.print_exc()
                    continue   # out of retries, continue
                time.sleep(3)   # wait 3 seconds before next try


        # ------------------ Wrap episode processing in a try-except to handle lightning strikes ------------------
        rollout_size = 10
        correct_rollout = []
        for update in range(rollout_size):
            epoch_idx = epoch_idx[torch.randperm(len(epoch_idx))]  # Shuffle the indices
            num_batches = (len(epoch_idx) + batch_size - 1) // batch_size
            for batch in range(num_batches):
                batch_idx = epoch_idx[batch * batch_size : (batch+1) * batch_size]
                for attempt in range(1, max_attempts + 1):
                    try:
                        # -------- Process Episodes and Backprop in Minibatches --------
                        mb_size = process_minibatch_size if process_minibatch_size is not None else len(batch_idx)
                        num_minibatches = (len(batch_idx) + mb_size - 1) // mb_size
                        for mb in range(num_minibatches):
                            # Slice minibatch
                            mb_idx = batch_idx[mb * mb_size : (mb + 1) * mb_size]
                            alphabet_states_minibatch = alphabet_states_batch[mb_idx]
                            guess_states_minibatch = guess_states_batch[mb_idx]
                            expected_values_minibatch = expected_values_batch[mb_idx]
                            expected_rewards_minibatch = expected_rewards_batch[mb_idx]
                            rewards_minibatch = rewards_batch[mb_idx]
                            guess_mask_minibatch = guess_mask_batch[mb_idx]
                            active_mask_minibatch = active_mask_batch[mb_idx]
                            valid_mask_minibatch = valid_mask_batch[mb_idx]

                            # ---------------- Process Episodes Old Net ----------------
                            # with torch.no_grad():
                            #     _, old_probs_minibatch, _, _ = process_episodes(
                            #         old_policy_net,
                            #         alphabet_states_minibatch,
                            #         guess_states_minibatch,
                            #         expected_values_minibatch,
                            #         expected_rewards_minibatch,
                            #         rewards_minibatch,
                            #         active_mask_minibatch,
                            #         valid_mask_minibatch,
                            #         alpha,
                            #         temperature,
                            #         gamma,
                            #         lam,
                            #         reward_blend_factor,
                            #         value_blend_factor,
                            #     )

                            # ---------------- Process Episodes Best Checkpoint ----------------
                            # with torch.no_grad():
                            #     _, best_probs_minibatch, _, _ = process_episodes(
                            #         best_policy_net,
                            #         alphabet_states_minibatch,
                            #         guess_states_minibatch,
                            #         expected_values_minibatch,
                            #         expected_rewards_minibatch,
                            #         rewards_minibatch,
                            #         active_mask_minibatch,
                            #         valid_mask_minibatch,
                            #         alpha,
                            #         temperature,
                            #         gamma,
                            #         lam,
                            #         reward_blend_factor,
                            #         value_blend_factor,
                            #     )

                            # ---------------- Process Episodes Current Net ----------------
                            actor_critic_net.train()
                            advantages_minibatch, probs_minibatch, guide_probs_minibatch, correct_minibatch = process_episodes(
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
                                reward_blend_factor,
                                value_blend_factor,
                            )
                            old_probs_minibatch = probs_minibatch.detach()  # Detach old probs to prevent gradients from flowing back to the old policy
                            best_probs_minibatch = probs_minibatch.detach()  # Detach best probs to prevent gradients from flowing back to the best policy

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
                                clip_probs=True,
                                clip_advantages=False,
                            )

                            # ------------------ Normalize Minibatch Losses ------------------
                            # Batch loss should be invariant to minibatch size
                            mb_proportion = (len(mb_idx) / len(batch_idx))
                            loss_mb = loss_mb * mb_proportion
                            actor_loss_mb = actor_loss_mb * mb_proportion
                            critic_loss_mb = critic_loss_mb * mb_proportion
                            entropy_loss_mb = entropy_loss_mb * mb_proportion
                            kl_reg_loss_mb = kl_reg_loss_mb * mb_proportion
                            kl_guide_loss_mb = kl_guide_loss_mb * mb_proportion
                            kl_best_loss_mb = kl_best_loss_mb * mb_proportion

                            # ---------------- Measure Grad Norms ----------------
                            if (epoch%1 == 0) and (batch == num_batches - 1) and (mb == num_minibatches - 1) and (update == rollout_size - 1):  # Measure grad norms on the last minibatch of the last update of the last batch of every epochs
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
                            if update == rollout_size - 1:  # Only accumulate results on the last update of the rollout (the correct values will be the same for all updates)
                                correct_rollout.append(correct_minibatch)
                            loss_mb.backward()

                            # ---------------- Rest Computer to Prevent Overheating ----------------
                            rest_computer(len(mb_idx))  # Rest based on minibatch size

                        # -------------- Backprop --------------
                        torch.nn.utils.clip_grad_norm_(actor_critic_net.parameters(), max_norm=3.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()

                        break
                    # ---------------- Print exception errors and continue ----------------
                    except RuntimeError as e:
                        if attempt < max_attempts:
                            print(f"Retrying batch {batch} due to error:\n")
                            traceback.print_exc()
                        else:
                            optimizer.zero_grad(set_to_none=True)  # Reset optimizer state
                            print(f"Failed to process batch {batch} after {max_attempts} attempts due to error:\n.")
                            traceback.print_exc()
                            continue   # out of retries, continue
                        time.sleep(3)   # wait 3 seconds before next try

        # ---------------- Correct for a Given Policy ----------------
        correct_rollout = torch.cat(correct_rollout, dim=0)

        # ---------------- Update Replay ----------------
        replay_loader.update(epoch_idx, epoch_idx[~correct_rollout.cpu()])

        # ---------------- Evaluate Learning on Full Vocab ----------------
        target_idx = torch.arange(len(target_vocab)).to(device)
        size1 = collect_minibatch_size if collect_minibatch_size is not None else len(target_idx)
        size2 = process_minibatch_size if process_minibatch_size is not None else len(target_idx)
        test_mb_size = min(size1, size2)
        (test_loss, test_actor_loss, test_critic_loss, test_entropy_loss, test_kl_reg_loss, test_kl_guide_loss, test_kl_best_loss, test_correct, test_accuracy, test_guesses,) = test(
            old_policy_net,
            best_policy_net,
            actor_critic_net,
            total_vocab,
            target_vocab,
            max_guesses,
            target_idx,
            test_mb_size,
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
            reward_blend_factor,
            value_blend_factor,
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
            f"Entropy Loss: {test_entropy_loss:.4f} * {entropy_coef:.2f}, Train KL-Reg Loss: {test_kl_reg_loss:.4f} * {kl_reg_coef:.2f}, "
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
                save_checkpoint(actor_critic_net, test_accuracy, test_guesses, config, log_dir, name='best_model.pth')

        # ----------------- Update Statistics ----------------
        best_accuracy = max(test_accuracy, best_accuracy)
        best_guesses = min(test_guesses, best_guesses)
        best_test_actor_loss = min(test_actor_loss, best_test_actor_loss)
        best_test_critic_loss = min(test_critic_loss, best_test_critic_loss)

        # ---------------- Early Stopping ----------------
        if no_improve_count >= early_stopping_patience:
            print(f'No improvement for {no_improve_count} epochs, stopping training.')
            if checkpointing:
                save_checkpoint(actor_critic_net, test_accuracy, test_guesses, config, log_dir, name='pretrained_model.pth')
            break

        # ---------------- Overall Stopping ----------------
        if epoch == epochs:
            print(f'Stopping training at epoch {epochs}.')
            if checkpointing:
                save_checkpoint(actor_critic_net, test_accuracy, test_guesses, config, log_dir, name='pretrained_model.pth')
            break
            
    print('Pretraining complete!')



############################################
# POSTTRAINING LOOP
############################################
def posttrain(
    actor_critic_net,
    total_vocab,
    target_vocab,
    max_guesses,
    lr,
    batch_size,
    collect_minibatch_size,
    process_minibatch_size,
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
    warmup_steps,
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

    replay_loader = HardWordBuffer(target_vocab, capacity=len(target_vocab), replay_ratio=2.0, rho=0.03)

    replay=True
    optimizer = optim.AdamW(actor_critic_net.parameters(), lr=lr, weight_decay=1e-4)
    warmup_factor = 1.0 / warmup_steps
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, end_factor=1.0, total_iters=warmup_steps)

    min_lr = lr * min_lr_factor

    best_test_actor_loss = float('inf')
    best_test_critic_loss = float('inf')
    best_accuracy = 0.0
    best_guesses = float(max_guesses)
    no_improve_count = 0

    old_policy_net = copy.deepcopy(actor_critic_net).eval().to(device)
    for p in old_policy_net.parameters():
        p.requires_grad = False
    best_policy_net = copy.deepcopy(actor_critic_net).eval().to(device)
    for p in best_policy_net.parameters():
        p.requires_grad = False
    actor_critic_net.train()

    reward_blend_factor = 0.9
    value_blend_factor = 0.1

    max_attempts = 3

    for epoch in range(1, epochs + 1):
        # ------------------- Generate Epoch Indices -------------------
        # target_vocab_idx = torch.randperm(len(target_vocab)).to(device)
        target_vocab_idx = torch.cat([torch.arange(len(target_vocab)) for _ in range(3)]).to(device)
        if replay:
            replay_idx = torch.tensor(replay_loader.sample()).to(device)
            epoch_idx = torch.cat((target_vocab_idx, replay_idx), dim=0)  # [(1 + replay_ratio) * len(target_vocab)]
            epoch_idx = epoch_idx[torch.randperm(len(epoch_idx))]
        else:
            epoch_idx = target_vocab_idx

        # ------------------ Refresh Network for New Rollout ------------------
        old_policy_net.load_state_dict(actor_critic_net.state_dict())

        # ------------------ Wrap episode collection in a try-except to handle lightning strikes ------------------
        for attempt in range(1, max_attempts + 1):
            try:
                # -------- Collect Episodes in Minibatches --------
                alphabet_states_epoch, guess_states_epoch, expected_values_epoch, expected_rewards_epoch, rewards_epoch, guess_mask_epoch, active_mask_epoch, valid_mask_epoch = [], [], [], [], [], [], [], []
                mb_size = collect_minibatch_size if collect_minibatch_size is not None else len(epoch_idx)
                num_minibatches = (len(epoch_idx) + mb_size - 1) // mb_size
                for mb in range(num_minibatches):
                    # Slice minibatch
                    mb_idx = epoch_idx[mb * mb_size : (mb + 1) * mb_size]
                    target_tensor = target_vocab_tensor[mb_idx]
                    # Collect minibatch episodes
                    (alphabet_states_minibatch, guess_states_minibatch, _, expected_values_minibatch, expected_rewards_minibatch, rewards_minibatch, guess_mask_minibatch, active_mask_minibatch, valid_mask_minibatch) = collect_episodes(
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
                    # Append to batch lists
                    alphabet_states_epoch.append(alphabet_states_minibatch)
                    guess_states_epoch.append(guess_states_minibatch)
                    expected_values_epoch.append(expected_values_minibatch)
                    expected_rewards_epoch.append(expected_rewards_minibatch)
                    rewards_epoch.append(rewards_minibatch)
                    guess_mask_epoch.append(guess_mask_minibatch)
                    active_mask_epoch.append(active_mask_minibatch)
                    valid_mask_epoch.append(valid_mask_minibatch)

                # Concatenate Batch Results
                alphabet_states_epoch = torch.cat(alphabet_states_epoch, dim=0)  # [epoch_size, max_guesses+1, 26, 11]
                guess_states_epoch = torch.cat(guess_states_epoch, dim=0)  # [epoch_size, max_guesses+1, 26, 11]
                expected_values_epoch = torch.cat(expected_values_epoch, dim=0)  # [epoch_size, max_guesses+1]
                expected_rewards_epoch = torch.cat(expected_rewards_epoch, dim=0)  # [epoch_size, max_guesses+1]
                rewards_epoch = torch.cat(rewards_epoch, dim=0)  # [epoch_size, max_guesses+1]
                guess_mask_epoch = torch.cat(guess_mask_epoch, dim=0)  # [epoch_size, max_guesses+1, vocab_size]
                active_mask_epoch = torch.cat(active_mask_epoch, dim=0)  # [epoch_size, max_guesses+1]
                valid_mask_epoch = torch.cat(valid_mask_epoch, dim=0)  # [epoch_size, max_guesses+1]

                break
            # ---------------- Print exception errors and continue ----------------
            except RuntimeError as e:
                if attempt < max_attempts:
                    print(f"Retrying episode collection due to error:\n")
                    traceback.print_exc()
                else:
                    print(f"Failed to collects episodes after {max_attempts} attempts due to error:\n.")
                    clear_cache()
                    traceback.print_exc()
                    continue   # out of retries, continue
                time.sleep(3)   # wait 3 seconds before next try

        # ---------------- Multiple Passes Per Rollout ----------------
        rollout_size = 5
        for update in range(rollout_size):
        # ------------------ Wrap episode pass-through in a try-except to handle lightning strikes ------------------
            for attempt in range(1, max_attempts + 1):
                try:
                    correct_rollout = []
                    rollout_idx = torch.randperm(len(epoch_idx))  # Shuffle the indices
                    b_size = batch_size if batch_size is not None else len(epoch_idx)
                    num_batches = (len(rollout_idx) + b_size - 1) // b_size
                    # ------------------ Pass Through Episodes ------------------
                    for batch in range(num_batches):
                        batch_idx = rollout_idx[batch * b_size : (batch+1) * b_size]
                        # -------- Process Episodes and Backprop in Minibatches --------
                        mb_size = process_minibatch_size if process_minibatch_size is not None else len(batch_idx)
                        num_minibatches = (len(batch_idx) + mb_size - 1) // mb_size
                        for mb in range(num_minibatches):
                            # Slice minibatch
                            mb_idx = batch_idx[mb * mb_size : (mb + 1) * mb_size]
                            alphabet_states_minibatch = alphabet_states_epoch[mb_idx]
                            guess_states_minibatch = guess_states_epoch[mb_idx]
                            expected_values_minibatch = expected_values_epoch[mb_idx]
                            expected_rewards_minibatch = expected_rewards_epoch[mb_idx]
                            rewards_minibatch = rewards_epoch[mb_idx]
                            guess_mask_minibatch = guess_mask_epoch[mb_idx]
                            active_mask_minibatch = active_mask_epoch[mb_idx]
                            valid_mask_minibatch = valid_mask_epoch[mb_idx]

                            # ---------------- Process Episodes Old Policy ----------------
                            with torch.no_grad():
                                _, old_probs_minibatch, _, _ = process_episodes(
                                    old_policy_net,
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
                                    reward_blend_factor,
                                    value_blend_factor,
                                )

                            # ---------------- Process Episodes Best Checkpoint ----------------
                            with torch.no_grad():
                                _, best_probs_minibatch, _, _ = process_episodes(
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
                                    reward_blend_factor,
                                    value_blend_factor,
                                )

                            # ---------------- Process Episodes Current Net ----------------
                            actor_critic_net.train()
                            advantages_minibatch, probs_minibatch, guide_probs_minibatch, correct_minibatch = process_episodes(
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
                                reward_blend_factor,
                                value_blend_factor,
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
                                clip_probs=False,
                                clip_advantages=True,
                            )

                            # ------------------ Normalize Minibatch Losses ------------------
                            mb_proportion = (len(mb_idx) / len(batch_idx))  # make batch loss invariant to minibatch size
                            loss_mb = loss_mb * mb_proportion
                            actor_loss_mb = actor_loss_mb * mb_proportion
                            critic_loss_mb = critic_loss_mb * mb_proportion
                            entropy_loss_mb = entropy_loss_mb * mb_proportion
                            kl_reg_loss_mb = kl_reg_loss_mb * mb_proportion
                            kl_guide_loss_mb = kl_guide_loss_mb * mb_proportion
                            kl_best_loss_mb = kl_best_loss_mb * mb_proportion

                            # ---------------- Measure Grad Norms ----------------
                            if (epoch%3 == 0) and (batch == num_batches - 1) and (mb == num_minibatches - 1) and (update == rollout_size-1):  # Measure grad norms on one minibatch in the rollout
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
                            if update == rollout_size - 1:  # Only accumulate results on the last update of the rollout (the correct values will be the same for all updates)
                                correct_rollout.append(correct_minibatch)
                            loss_mb.backward()

                            # ---------------- Rest Computer to Prevent Overheating ----------------
                            rest_computer(len(mb_idx))  # Rest based on minibatch size

                        # -------------- Backprop --------------
                        torch.nn.utils.clip_grad_norm_(actor_critic_net.parameters(), max_norm=3.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()

                    break
                # ---------------- Print exception errors and continue ----------------
                except RuntimeError as e:
                    if attempt < max_attempts:
                        print(f"Retrying episode processing due to error:\n")
                        traceback.print_exc()
                    else:
                        optimizer.zero_grad(set_to_none=True)  # Reset optimizer state
                        clear_cache()
                        print(f"Failed to process episodes after {max_attempts} attempts due to error:\n.")
                        traceback.print_exc()
                        continue   # out of retries, continue
                    time.sleep(3)   # wait 3 seconds before next try

        # ---------------- Correct for a Given Policy ----------------
        correct_rollout = torch.cat(correct_rollout, dim=0)
        rollout_accuracy = torch.mean(correct_rollout.float())

        # ---------------- Update Replay ----------------
        rollout_target_idx = epoch_idx[rollout_idx]
        replay_loader.update(rollout_target_idx, rollout_target_idx[~correct_rollout.cpu()])

        # ---------------- Evaluate Learning on Full Vocab ----------------
        target_idx = torch.arange(len(target_vocab)).to(device)
        mb_size1 = collect_minibatch_size if collect_minibatch_size is not None else len(target_idx)
        mb_size2 = process_minibatch_size if process_minibatch_size is not None else len(target_idx)
        test_mb_size = min(mb_size1, mb_size2)
        (test_loss, test_actor_loss, test_critic_loss, test_entropy_loss, test_kl_reg_loss, test_kl_guide_loss, test_kl_best_loss, test_correct, test_accuracy, test_guesses,) = test(
            old_policy_net,
            best_policy_net,
            actor_critic_net,
            total_vocab,
            target_vocab,
            max_guesses,
            target_idx,
            test_mb_size,
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
            reward_blend_factor,
            value_blend_factor,
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
            f"Epoch {epoch}/{epochs} | Test Acc: {test_accuracy:.2%} | Test Avg Guesses: {test_guesses:.2f} | "
            f"Rollout Accuracy: {rollout_accuracy:.2%} | alpha={alpha:.2f}, temp={temperature:.2f} | "
            f"Actor Loss: {test_actor_loss:.4f} * {actor_coef:.2f}, Critic Loss: {test_critic_loss:.4f} * {critic_coef:.2f}| "
            f"Entropy Loss: {test_entropy_loss:.4f} * {entropy_coef:.2f}, Train KL-Reg Loss: {test_kl_reg_loss:.4f} * {kl_reg_coef:.2f}, "
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
        if (no_improve_count >= greedify_patience):
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

        # ----------------- Update Statistics ----------------
        best_accuracy = max(test_accuracy, best_accuracy)
        best_guesses = min(test_guesses, best_guesses)
        best_test_actor_loss = min(test_actor_loss, best_test_actor_loss)
        best_test_critic_loss = min(test_critic_loss, best_test_critic_loss)

        # ---------------- Early Stopping ----------------
        if no_improve_count >= early_stopping_patience:
            print(f'No improvement for {no_improve_count} epochs, stopping training.')
            if checkpointing:
                save_checkpoint(actor_critic_net, test_accuracy, test_guesses, config, log_dir, name='posttrained_model.pth')
            break

        # ---------------- Overall Stopping ----------------
        if epoch == epochs: 
            print(f'Stopping training at epoch {epochs}.')
            if checkpointing:
                save_checkpoint(actor_critic_net, test_accuracy, test_guesses, config, log_dir, name='posttrained_model.pth')
            break

        # ---------------- Rest Computer to Prevent Overheating ----------------
        rest_computer(len(target_vocab))  # Rest based on target vocab length
            
    print('Training complete!')



############################################
# TESTING LOOP
############################################
def test(
    old_policy_net,
    best_policy_net,
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
    gamma,
    lam,
    m,
    reward_blend_factor,
    value_blend_factor,
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
    test_loss = test_actor_loss = test_critic_loss = test_entropy_loss = test_kl_reg_loss = test_kl_guide_loss = test_kl_best_loss = 0.0
    test_guesses = test_samples = 0.0
    test_correct = []
    # Dummy exploration variables because we have argmax=True
    alpha = 0.0
    temperature = 1.00

    old_policy_net.eval()
    best_policy_net.eval()
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

                    # ---------------- Process Episodes Old Net ----------------
                    _, old_probs_minibatch, _, _ = process_episodes(
                        old_policy_net,
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
                        reward_blend_factor,
                        value_blend_factor,
                    )

                    # ---------------- Process Episodes Best Checkpoint ----------------
                    _, best_probs_minibatch, _, _ = process_episodes(
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
                        reward_blend_factor,
                        value_blend_factor,
                    )

                    # ---------------- Process Episodes ----------------
                    advantages_minibatch, probs_minibatch, guide_probs_minibatch, correct_minibatch = process_episodes(
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
                        reward_blend_factor,
                        value_blend_factor,
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
                        clip_probs=False,
                        clip_advantages=False,
                    )

                    # ------------------ Accumulate Minibatch Results ------------------
                    mb_proportion = (len(mb_idx) / len(target_idx))                    
                    test_loss = test_loss + loss_mb.item() * mb_proportion
                    test_actor_loss = test_actor_loss + actor_loss_mb.item() * mb_proportion
                    test_critic_loss = test_critic_loss + critic_loss_mb.item() * mb_proportion
                    test_entropy_loss = test_entropy_loss + entropy_loss_mb.item() * mb_proportion
                    test_kl_reg_loss = test_kl_reg_loss + kl_reg_loss_mb.item() * mb_proportion
                    test_kl_guide_loss = test_kl_guide_loss + kl_guide_loss_mb.item() * mb_proportion
                    test_kl_best_loss = test_kl_best_loss + kl_best_loss_mb.item() * mb_proportion
                    test_correct.append(correct_minibatch)
                    test_guesses += (active_mask_minibatch[:, :-1].sum(dim=-1).sum(dim=-1)).item()
                    test_samples += len(mb_idx)  # total number of samples

                    # ---------------- Rest Computer to Prevent Overheating ----------------
                    rest_computer(len(mb_idx))  # Rest based on minibatch size

                break
            # ---------------- Print exception errors and continue ----------------
            except RuntimeError as e:
                if attempt < max_attempts:
                    print(f"Retrying test loop due to error:\n")
                    traceback.print_exc()
                else:
                    print(f"Failed to execute test loop after {max_attempts} attempts due to error:\n.")
                    traceback.print_exc()
                    continue   # out of retries, continue
                time.sleep(3)   # wait 3 seconds before next try

    # ---------------- Concatenate minibatch results ----------------
    test_correct = torch.cat(test_correct, dim=0)

    # Calculate metrics
    test_accuracy = (test_correct.sum().item() / test_samples)
    test_guesses = (test_guesses / test_samples)  # average guesses per sample

    return (
        test_loss,
        test_actor_loss,
        test_critic_loss,
        test_entropy_loss,
        test_kl_reg_loss,
        test_kl_guide_loss,
        test_kl_best_loss,
        test_correct,
        test_accuracy,
        test_guesses,
    )
