import math
import numpy as np
import torch
import torch.nn.functional as F
from simulation_utils import make_probs, wordle_step


##############################################################################
#                            MCTS Node Class
##############################################################################


class MCTSNode:
    """
    Represents a single node in the MCTS tree for one environment/state.
    """

    def __init__(
        self,
        alphabet_state,
        guess_state,
        guess_num,
        guess_word,
        target_word,
        parent=None,
        prior=0.0,
        reward=0.0,
    ):
        """
        Args:
            state: A Python object or tensor that represents the environment state.
            parent: The parent node in the tree, or None if this is the root.
            prior: The prior probability P(a|s) from the policy network for this action.
            is_terminal_fn: A function that checks if 'state' is terminal (optional).
        """
        self.alphabet_state = alphabet_state  # shape [1, 26, 11]
        self.guess_state = guess_state  # shape [1, max_guesses]
        self.state = torch.cat([alphabet_state.view(1, -1), guess_state], dim=-1)  # shape [1, 26*11 + max_guesses]
        self.guess_num = guess_num
        self.guess_word = guess_word
        self.target_word = target_word
        self.parent = parent
        self.prior = prior  # from the policy's prior probability
        self.reward = reward  # immediate reward from the environment

        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action -> MCTSNode

    @property
    def value_avg(self):
        """Average node value based on sum of backpropagated rollouts/values."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_terminal(self):
        return (self.guess_num >= 6) or (self.guess_word == self.target_word)

    def expand(self, actor_critic_net, vocab, alpha, temperature, top_k=50):
        """
        Create children for this node by querying the policy network on this node's state.

        Args:
            actor_critic_net: model that will return (logits, value).
            vocab: List of all possible actions (words).
            alpha, temperature: Exploration parameters for the policy.
            top_k: How many actions from the policy to expand to children.
        """
        if self.is_terminal():
            return  # no need to expand

        # 1) Get probs
        with torch.no_grad():
            logits, _ = actor_critic_net(self.state)  # shape=[1, vocab_size]
            probs = make_probs(logits, alpha, temperature)  # shape=[1, vocab_size]

        # 2) Get top-K indices
        if top_k < probs.size(-1):
            _, top_indices = torch.topk(probs, top_k, dim=-1)  # shape=[1, top_k]
            top_indices = top_indices.squeeze(0)  # shape=[top_k]
        else:
            top_indices = torch.arange(probs.size(-1))  # shape=[vocab_size]
        # 3) Create a child node for each top-K action
        for idx in top_indices:
            action_prior = probs[0, idx.item()]  # shape: scalar
            target_word = self.target_word
            guess_word = vocab[idx]
            new_alphabet_state, new_guess_state, reward, correct = wordle_step(self.alphabet_state, self.guess_state, guess_word, target_word)
            child = MCTSNode(
                alphabet_state=new_alphabet_state,
                guess_state=new_guess_state,
                guess_num=self.guess_num + 1,
                guess_word=guess_word,
                target_word=target_word,
                parent=self,
                prior=action_prior,
                reward=reward,
            )
            self.children[idx] = child

    def evaluate(self, actor_critic_net, gamma=1.0):
        """
        Estimate the node's value using the critic and the immediate reward.
        """
        # Can do rollout instead later
        if self.is_terminal():
            # The final reward is the immediate reward if we guessed the word or ran out of guesses
            return self.reward
        else:
            with torch.no_grad():
                _, value = actor_critic_net(self.state)  # shape=[1,1]
            return self.reward + gamma * value.item()


##############################################################################
#                           MCTS Utilities
##############################################################################


def select_child_via_ucb(node, c_puct=1.0):
    """
    Select the child with the highest UCB score:
      UCB = Q + c_puct * prior * sqrt(parent_visits) / (1 + child.visit_count)
    """
    best_score = -float('inf')
    best_child = None

    parent_visits = max(1, node.visit_count)
    sum_visits = sum(child.visit_count for child in node.children.values()) + 1e-8

    for action_idx, child in node.children.items():
        q_value = child.value_avg
        u_value = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
        score = q_value + u_value
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def rollout_or_value_estimate(node, actor_critic_net):
    """
    Simple function to return a value estimate for a node.
    Could be a full environment rollout or a direct critic call.
    """
    value_est = node.evaluate(actor_critic_net)
    return value_est


def backpropagate(path, value_est):
    """
    Accumulate visits and update the Q-value along the path.
    path: list of nodes from root to leaf.
    value_est: the final value from rollout/critic.
    """
    for node in path:
        node.visit_count += 1
        node.value_sum += value_est


##############################################################################
#                               MCTS Search
##############################################################################
def mcts_search(
    root_alphabet_state,
    root_guess_state,
    guess_num,
    target_word,
    actor_critic_net,
    vocab,
    alpha,
    temperature,
    num_simulations=30,
    top_k=30,
    c_puct=1.0,
):
    """
    Run MCTS starting from the given root state for a single environment
    (i.e., one puzzle) and pick the best action (child).

    Args:
        root_alphabet_state: Tensor [1, 26, 11] for Wordle's letter info at the root.
        root_guess_state:    Tensor [1, max_guesses] for which guess number is active.
        guess_num:           Int, how many guesses used so far.
        target_word:         String, the correct Wordle solution.
        actor_critic_net:    Model returning (logits, value).
        vocab:               List of possible guess words (actions).
        alpha, temperature:  For 'make_probs' inside MCTS node expansion.
        num_simulations:     How many times to run the MCTS loop.
        top_k:               Expand top-K children at each node (helps limit branching).
        c_puct:              Exploration constant for the UCB formula.

    Returns:
        best_action_idx (int): index into 'vocab' of the best action from the root.
    """
    # 1) Create the root node
    root_node = MCTSNode(
        alphabet_state=root_alphabet_state,
        guess_state=root_guess_state,
        guess_num=guess_num,
        target_word=target_word,
        parent=None,
        prior=1.0,
        reward=0.0,
    )
    # Expand it once to get initial children
    root_node.expand(actor_critic_net, vocab, alpha, temperature, top_k=top_k)

    # 2) Run repeated simulations
    for _ in range(num_simulations):
        node = root_node
        path = [node]

        # --- SELECTION ---
        # Descend while node has children and is not terminal
        while node.children and not node.is_terminal():
            node = select_child_via_ucb(node, c_puct=c_puct)
            path.append(node)

        # --- EXPANSION ---
        if not node.is_terminal():
            node.expand(actor_critic_net, vocab, alpha, temperature, top_k=top_k)

        # --- ROLLOUT / VALUE ESTIMATE ---
        value_est = rollout_or_value_estimate(node, actor_critic_net)

        # --- BACKPROP ---
        backpropagate(path, value_est)

    # 3) Choose the action with the highest visit count from root
    best_child, best_action_idx, best_visits = None, None, -1
    for action_idx, child in root_node.children.items():
        if child.visit_count > best_visits:
            best_visits = child.visit_count
            best_child = child
            best_action_idx = action_idx

    return best_action_idx


# def select_actions_with_mcts(
#     actor_critic_net,
#     batch_alphabet_states,
#     batch_guess_states,
#     batch_guess_nums,
#     batch_target_words,
#     vocab,
#     alpha,
#     temperature,
#     num_simulations=30,
#     top_k=50,
#     c_puct=1.0
# ):
#     # we do a loop over batch_size, calling MCTS for each environment,
#     # then return the chosen guess index for each environment
#     batch_size = len(batch_alphabet_states)
#     guess_indices = []

#     for i in range(batch_size):
#         best_idx = mcts_search(
#             root_alphabet_state=batch_alphabet_states[i],
#             root_guess_state=batch_guess_states[i],
#             guess_num=batch_guess_nums[i],
#             target_word=batch_target_words[i],
#             actor_critic_net=actor_critic_net,
#             vocab=vocab,
#             alpha=alpha,
#             temperature=temperature,
#             num_simulations=num_simulations,
#             top_k=top_k,
#             c_puct=c_puct
#         )
#         guess_indices.append(best_idx)

#     guess_words = [vocab[idx] for idx in guess_indices]
#     return guess_indices, guess_words
