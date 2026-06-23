
        #     # Calculate new reward responses that align with best mean guesses
        #     # G = 6
        #     # H = 11.768
        #     # solved = (
        #     #     episodes["responses"]["correct"].bool() & episodes["states"]["active_mask"][:, :-1, ...].bool()
        #     # ).any(dim=1).float()
        #     # total_entropy_reward = (H - episodes["states"]["entropy"][:, -1, ...]) / H
        #     # num_guesses = episodes["states"]["active_mask"][:, :-1, ...].float().sum(dim=1)
        #     # guess_num_reward = solved * ((G - num_guesses) / G)
        #     # correct_reward = 2.0 * solved
        #     # episodes["responses"]["new_rewards"] = (
        #     #     total_entropy_reward + guess_num_reward + correct_reward
        #     # ).unsqueeze(dim=1).repeat(1, G, 1)

        #     G = 6
        #     gamma = 1.0
        #     active = episodes["states"]["active_mask"][:, :-1, ...].float()
        #     entropy_reward = episodes["responses"]["expected_rewards"] * active
        #     # naive_new_rewards = entropy_reward - (active / G) + (2.0 * episodes["responses"]["correct"].float() * active)
        #     naive_new_rewards = entropy_reward - (active / G)

        #     # Do gamma backtracking
        #     returns = torch.zeros_like(naive_new_rewards)
        #     running = 0.0
        #     for t in reversed(range(naive_new_rewards.shape[1])):
        #         running = naive_new_rewards[:, t] + gamma * running
        #         returns[:, t] = running
        #     episodes["responses"]["new_rewards"] = returns



        # # Std adv is global to not bias towards any hidden state or timestep
        # adv_diff = (advantages - adv_mean) * active_mask                                # [B, G, R]
        # adv_sum_square = (adv_diff.pow(2)).sum(dim=[0, 1, 2], keepdim=True)             # [1, 1, 1]
        # adv_std_count = active_mask.sum(dim=[0, 1, 2], keepdim=True).clamp_min(1.0)     # [1, 1, 1]
        # adv_std = (adv_sum_square / adv_std_count).sqrt().clamp_min(self.eps)           # [1, 1, 1]