import torch
import torch.nn.functional as F
import math
import string


letter_to_idx = {ch: i for i, ch in enumerate(string.ascii_lowercase)}

def words_to_tensor(words):
    """
    Convert a list of words (lowercase, length=5)
    - words_tensor: shape [batch_size, 5]
    """
    mapped = []
    for w in words:
        mapped.append([letter_to_idx[ch] for ch in w])
    words_tensor = torch.tensor(mapped, dtype=torch.long).view(-1, 5)
    return words_tensor


############################################
# STATE UTILS
############################################
def update_alphabet_states(given_alphabet_states, guess_words, target_words):
    """
    Update the alphabet state in tensor operations. Specifically,

    Inputs:
    - given_alphabet_state: shape [batch_size, 26, 11]
    - guess_word: list of strings of length 5
    - target_word: list of strings of length 5

    Word --> Tensor:
    - guess_tensor: converted letters to numbers -- shape [batch_size, 26, 5]
    - target_tensor: converted letters to numbers -- shape [batch_size, 26, 5]
    - alphabet_tensor: 1-26 repeated 5 times -- shape [batch_size, 26, 5]
    - guess_loc: boolean mask for guess letter locations -- shape [batch_size, 26, 5]
    - target_loc: boolean mask for target letter locations -- shape [batch_size, 26, 5]

    Masks:
    - guessed: number of times a letter appears in our guessed word -- shape [batch_size, 26, 1]
    - exist: number of times a letter appears in the target word -- shape [batch_size, 26, 1]
    - letter_count: minimum of guessed and exist. We know a letter must appear this many times in the target! -- shape [batch_size, 26, 1]
    - green_mask: boolean mask for correct letter guesses AND placement -- shape [batch_size, 26, 5]
    - not_exist: boolean mask for guessing a letter that is not in the target anywhere -- shape [batch_size, 26, 5]
    - found_not: boolean mask for guessing a letter that is in the target but not in the location -- shape [batch_size, 26, 5]
    - grey_mask: boolean mask for incorrect letter guesses -- shape [batch_size, 26, 5]

    Returns:
    - new_alphabet_state: updated [batch_size, 26, 11] tensor
    """
    batch_size = given_alphabet_states.size(0)

    # Tensorize guess/target information
    guess_tensor = words_to_tensor(guess_words).unsqueeze(1).repeat(1, 26, 1)  # shape [batch_size, 26, 5]
    target_tensor = words_to_tensor(target_words).unsqueeze(1).repeat(1, 26, 1)  # shape [batch_size, 26, 5]
    alphabet_tensor = torch.arange(0, 26).view(1, -1, 1).repeat(batch_size, 1, 5)  # shape [batch_size, 26, 5]
    guess_loc = guess_tensor == alphabet_tensor  # shape [batch_size, 26, 5]
    target_loc = target_tensor == alphabet_tensor  # shape [batch_size, 26, 5]

    # Calculate masks
    ## Counting occurrences
    guessed = guess_loc.sum(dim=-1, keepdim=True)  # shape [batch_size, 26, 1]
    exist = target_loc.sum(dim=-1, keepdim=True)  # shape [batch_size, 26, 1]
    letter_count = torch.min(guessed, exist)  # shape [batch_size, 26, 1]
    ## Checking matches
    green_mask = guess_loc & target_loc  # shape [batch_size, 26, 5]
    ## Checking non-matches
    not_exist = ((guessed > 0) & (exist == 0)).repeat(1, 1, 5)  # shape [batch_size, 26, 5]
    found_not = guess_loc & ~target_loc  # shape [batch_size, 26, 5]
    grey_mask = not_exist | found_not  # shape [batch_size, 26]

    # Combine and update
    guess_alphabet_states = torch.cat([letter_count, green_mask, grey_mask], dim=-1).float()
    new_alphabet_states = torch.max(guess_alphabet_states, given_alphabet_states)

    return new_alphabet_states


def calculate_alphabet_scores(alphabet_states):
    """
    Returns a weighted sum for letter scores, summed over the whole alphabet.
    - alphabet_states: representing the states of all letters [batch_size, 26, 11]
    - score: shape [batch_size]
    """
    # Define value weights
    weights = torch.cat(
        [
            torch.tensor([0.0]),  # Number of known occurences
            torch.tensor([0.0]).repeat(5),  # Matching letter and placement (greens)
            torch.tensor([0.0]).repeat(5),  # Unmatching letter and placement (yellows / greys)
        ]
    ).view(
        1, 1, -1
    )  # shape [1, 1, 11]

    # Calculate score
    weighted_state = alphabet_states * weights  # [batch_size, 26, 11]
    scores = weighted_state.sum(dim=-1).sum(dim=-1)  # sum over letters (dim=1) and columns (dim=2)  # shape [batch_size]

    return scores


def calculate_rewards(new_alphabet_states, given_alphabet_states, guess_words, target_words):
    """
    Calculate the reward given two alphabet states.

    Inputs:
      new_alphabet_states: [batch_size, 26, 11]
      old_alphabet_states: [batch_size, 26, 11]
      guess_words: list of length batch_size (each a 5-char string)
      target_words: list of length batch_size (each a 5-char string)

    Output:
      rewards: [batch_size] float tensor
    """
    # Heuristic information reward with baseline penalty
    rewards = calculate_alphabet_scores(new_alphabet_states) - calculate_alphabet_scores(given_alphabet_states)  # shape [batch_size]

    # Bonus reward for correct guess
    correct_mask = torch.tensor(
        [1.0 if guess_words[i] == target_words[i] else 0.0 for i in range(len(guess_words))],
        dtype=torch.float32,
    )
    rewards += 1.0 * correct_mask  # shape [batch_size]

    return rewards


############################################
# SIMULATION UTILS
############################################
def make_probs(logits, alpha, temperature):
    """
    Returns the final probabilities after mixing with uniform distribution and temperature scaling.
    - logits: [batch_size, action_size]
    - alpha: float (uniform mixing parameter)
    - temperature: float (temperature for softmax)
    """
    # Softmax with temperature
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Uniform distribution for alpha-mixing
    uniform_probs = torch.ones_like(probs) / probs.size(-1)
    probs = alpha * uniform_probs + (1 - alpha) * probs

    return probs


def select_actions(actor_critic_net, alphabet_states, guess_states, guess_num, guess_mask_batch, target_words, vocab, alpha, temperature, argmax=False, k=1):
    """
    For each environment in the batch, compute probabilities, and select a word.
    - alphabet_states: [batch_size, 26, 11]
    - guess_states: [batch_size, max_guesses]
    - guess_num: [batch_size] (current guess number)
    - guess_mask_batch: [batch_size, max_guesses, action_size] (mask of already guessed words)
    - alpha, temperature: float (exploration parameters)
    """
    batch_size = guess_idx.size(0)
    action_size = len(vocab)

    # Forward pass to get logits and value
    states = torch.cat([alphabet_states.view(-1, 26 * 11), guess_states], dim=-1)  # shape: [batch_size, 26*11 + max_guesses]
    logits, values = actor_critic_net(states)  # shape: logits=[batch_size, action_size], values=[batch_size,1]
    probs = make_probs(logits, alpha, temperature)  # shape: [batch_size, action_size]

    # # Mask out already guessed words
    # if guess_mask_batch is not None:
    #     guessed = torch.any(guess_mask_batch, dim=1)  # shape: [batch_size, action_size]
    #     probs = probs * ~guessed  # Mask out already guessed words
    #     probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize

    # Sample actions
    if not argmax:
        guess_idx = torch.multinomial(probs, k)  # shape: [batch_size, k]
    else:
        _, guess_idx = torch.topk(probs, k, dim=-1)  # shape: [batch_size, k]
    guess_words = []
    for b in range(batch_size):
        row_indices = guess_idx[b]  # shape [k]
        row_words = [vocab[idx.item()] for idx in row_indices]
        guess_words.append(row_words)
    guess_idx = F.one_hot(guess_idx, num_classes=action_size).float()  # shape: [batch_size, action_size, k]

    return probs, values, guess_idx, guess_words


def wordle_step(alphabet_states, guess_states, guess_words, target_words):
    """
    Simulate one Wordle step:
      - alphabet_state: [batch_size, 26, 11]
      - guess_state: [batch_size, max_guesses]
      - guess_words: [batch_size, max_guesses]
      - target_words: [batch_size, max_guesses]

    Returns:
      - new_alphabet_states: updated [batch_size, 26, 11] tensor
      - new_guess_states: updated [batch_size, max_guesses] tensor
      - rewards: [batch_size] (based on discovered info + correct guess)
      - done: [batch_size] (bool)
    """
    batch_size = alphabet_states.size(0)

    new_alphabet_states = update_alphabet_states(alphabet_states, guess_words, target_words)
    new_guess_states = guess_states.roll(shifts=1, dims=1)

    rewards = calculate_rewards(new_alphabet_states, alphabet_states, guess_words, target_words)
    correct = torch.tensor([guess_words[i] == target_words[i] for i in range(batch_size)], dtype=torch.bool)

    return new_alphabet_states, new_guess_states, rewards, correct


def collect_episodes(actor_critic_net, vocab, target_words, alpha, temperature, max_guesses=6, argmax=False):
    """
    Collect a single Wordle episode with a *fixed length* rollout of 6 guesses.
    This will be masked to the actual number of guesses taken.

    Returns:
      alphabet_states: Tensor of shape [batch_size, max_guesses+1, 26, 11]
      guess_states: Tensor of shape [batch_size, max_guesses+1, max_guesses]
      log_probs: Tensor of shape [batch_size, max_guesses]
      rewards:  Tensor of shape [batch_size, max_guesses]
      done_mask: Tensor of shape [batch_size, max_guesses+1] (1 = valid step, 0 = after done)
    """
    # 1.0) Setup
    batch_size = len(target_words)
    action_size = len(vocab)

    # 1.1) Initialize Tensors
    alphabet_states_batch = torch.zeros((batch_size, max_guesses + 1, 26, 11), dtype=torch.float32)
    guess_states_batch = torch.zeros((batch_size, max_guesses + 1, max_guesses), dtype=torch.float32)
    policy_probs_batch = torch.zeros((batch_size, max_guesses, action_size), dtype=torch.float32)
    mcts_probs_batch = torch.zeros((batch_size, max_guesses, action_size), dtype=torch.float32)
    rewards_batch = torch.zeros((batch_size, max_guesses), dtype=torch.float32)
    guess_mask_batch = torch.zeros((batch_size, max_guesses, action_size), dtype=torch.bool)
    correct_mask_batch = torch.zeros((batch_size, max_guesses), dtype=torch.bool)
    active_mask_batch = torch.ones((batch_size, max_guesses + 1), dtype=torch.bool)
    guess_words_batch = []

    # 1.2) Initialize environment state
    alphabet_states = torch.zeros((batch_size, 26, 11), dtype=torch.float32)
    guess_states = torch.zeros([batch_size, max_guesses], dtype=torch.float32)
    guess_states[:, 0] = 1.0

    # 2) Roll out up to max_guesses
    active = torch.ones(batch_size, dtype=torch.bool)
    for guess_num in range(max_guesses):
        # Construct the state
        alphabet_states_batch[:, guess_num] = alphabet_states
        guess_states_batch[:, guess_num] = guess_states

        # # Select action
        # policy_probs, values, guess_idx, guess_words = select_actions(
        #     actor_critic_net, alphabet_states, guess_states, guess_num, guess_mask_batch, target_words, vocab, alpha, temperature, argmax
        # )

        # Select action with MCTS
        policy_probs, mcts_probs, values, guess_idx, guess_words = select_actions_mcts(
            actor_critic_net,
            alphabet_states,
            guess_states,
            guess_num,
            guess_mask_batch,
            target_words,
            vocab,
            alpha,
            temperature,
            argmax,
        )

        # Step environment
        alphabet_states, guess_states, rewards, correct = wordle_step(alphabet_states, guess_states, guess_words, target_words)
        active = active & (~correct)

        # Update
        policy_probs_batch[:, guess_num, :] = policy_probs
        mcts_probs_batch[:, guess_num, :] = mcts_probs
        rewards_batch[:, guess_num] = rewards
        guess_mask_batch[:, guess_num] = guess_idx
        guess_words_batch.append(guess_words)
        correct_mask_batch[:, guess_num] = correct
        active_mask_batch[:, guess_num + 1] = active

    active_mask_batch[:, -1] = 0  # Last step is not active

    return (
        alphabet_states_batch,
        guess_states_batch,
        rewards_batch,
        policy_probs_batch,
        mcts_probs_batch,
        guess_mask_batch,
        guess_words_batch,
        correct_mask_batch,
        active_mask_batch,
    )


def process_episodes(actor_critic_net, states_batch, rewards_batch, guess_mask_batch, correct_mask_batch, active_mask_batch, alpha, temperature, gamma, lam):
    """
    Process the collected episodes to compute advantages and final probabilities.

    Takes:
      states_batch:  [batch_size, max_guesses+1, state_size]
      rewards_batch: [batch_size, max_guesses]
      guess_mask_batch: [batch_size, max_guesses, action_size]
      correct_mask_batch: [batch_size, max_guesses]
      active_mask_batch: [batch_size, max_guesses+1]

    Returns:
      advantages_batch: [batch_size, max_guesses]
      probs_batch: [batch_size, max_guesses, action_size]
    """
    batch_size = states_batch.size(0)  # batch size
    max_guesses = states_batch.size(1) - 1  # max guesses (6 for Wordle)

    naive_logits, naive_values_batch = actor_critic_net(states_batch)  # shape: logits=[batch_size, max_guesses+1, action_size], value=[batch_size, max_guesses+1, 1]
    logits_batch = naive_logits[:, :-1, :]  # shape: [batch_size, max_guesses, action_size]
    values_batch = naive_values_batch.squeeze() * active_mask_batch.float()  # shape: [batch_size, max_guesses+1]
    rewards_batch = rewards_batch * active_mask_batch[:, :-1].float()  # shape: [batch_size, max_guesses]

    # Compute advantages for batch with uniform episode size
    advantages_batch = torch.zeros((batch_size, max_guesses), dtype=torch.float32)
    gae = torch.zeros(batch_size, dtype=torch.float32)
    for t in reversed(range(max_guesses)):
        delta = rewards_batch[:, t] + gamma * values_batch[:, t + 1] - values_batch[:, t]
        gae = delta + gamma * lam * gae
        advantages_batch[:, t] = gae

    # Calculate the final probs
    probs_batch = make_probs(logits_batch, alpha, temperature)

    return advantages_batch, probs_batch


def process_episodes_mcts(actor_critic_net, alphabet_states_batch, guess_states_batch, rewards_batch, guess_mask_batch, correct_mask_batch, active_mask_batch, alpha, temperature, gamma, lam):
    """
    Process the collected episodes to compute advantages and final probabilities.

    Takes:
      alphabet_states_batch:  [batch_size, max_guesses+1, 26, 11]
      guess_states_batch: [batch_size, max_guesses]
      rewards_batch: [batch_size, max_guesses]
      guess_mask_batch: [batch_size, max_guesses, action_size]
      correct_mask_batch: [batch_size, max_guesses]
      active_mask_batch: [batch_size, max_guesses+1]

    Returns:
      advantages_batch: [batch_size, max_guesses]
      probs_batch: [batch_size, max_guesses, action_size]
    """
    batch_size = alphabet_states_batch.size(0)  # batch size
    max_guesses = alphabet_states_batch.size(1) - 1  # max guesses (6 for Wordle)
    states_batch = torch.cat([alphabet_states_batch.view(batch_size, max_guesses+1, -1), guess_states_batch], dim=-1)  # shape: [batch_size, max_guesses+1, 26*11 + max_guesses]

    naive_logits, naive_values_batch = actor_critic_net(states_batch)  # shape: logits=[batch_size, max_guesses+1, action_size], value=[batch_size, max_guesses+1, 1]
    logits_batch = naive_logits[:, :-1, :]  # shape: [batch_size, max_guesses, action_size]
    values_batch = naive_values_batch.squeeze() * active_mask_batch.float()  # shape: [batch_size, max_guesses+1]
    rewards_batch = rewards_batch * active_mask_batch[:, :-1].float()  # shape: [batch_size, max_guesses]

    # Compute advantages for batch with uniform episode size
    advantages_batch = torch.zeros((batch_size, max_guesses), dtype=torch.float32)
    gae = torch.zeros(batch_size, dtype=torch.float32)
    for t in reversed(range(max_guesses)):
        delta = rewards_batch[:, t] + gamma * values_batch[:, t + 1] - values_batch[:, t]
        gae = delta + gamma * lam * gae
        advantages_batch[:, t] = gae

    # Calculate the final probs
    policy_probs_batch = make_probs(logits_batch, alpha, temperature)

    return advantages_batch, policy_probs_batch














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
        reward,
        guess_num,
        guess_word,
        guess_mask,
        target_word,
        parent=None,
        prior=0.0,
    ):
        """
        Args:
            state: A Python object or tensor that represents the environment state.
            parent: The parent node in the tree, or None if this is the root.
            prior: The prior probability P(a|s) from the policy network for this action.
            is_terminal_fn: A function that checks if 'state' is terminal (optional).
        """
        self.alphabet_state = alphabet_state.view(1, 26, 11)  # shape [1, 26, 11]
        self.guess_state = guess_state.view(1, -1)  # shape [1, max_guesses]
        self.state = torch.cat([alphabet_state.view(1, -1), guess_state.view(1, -1)], dim=-1)  # shape [1, 26*11 + max_guesses]
        self.reward = reward  # immediate reward from the environment
        self.guess_num = guess_num
        self.guess_word = guess_word
        self.guess_mask = guess_mask
        self.target_word = target_word
        self.parent = parent
        self.prior = prior  # from the policy's prior probability

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
        vocab_size = len(vocab)

        if self.is_terminal():
            return  # no need to expand

        # 1) Get probs
        with torch.no_grad():
            logits, _ = actor_critic_net(self.state)  # shape=[1, vocab_size]
            policy_probs = make_probs(logits, alpha, temperature)  # shape=[1, vocab_size]

        # Select actions

        # 2) Get top-K indices
        if top_k < vocab_size:
            _, top_idx = torch.topk(policy_probs, top_k, dim=-1)  # shape=[1, top_k]
            top_idx = top_idx.squeeze()  # shape=[top_k]
        else:
            top_idx = torch.arange(vocab_size).squeeze()  # shape=[vocab_size]
        # 3) Create a child node for each top-K action
        for idx in top_idx:
            action_prior = policy_probs[0, idx.item()]  # shape: scalar
            target_word = self.target_word
            guess_word = [vocab[idx]]
            new_alphabet_state, new_guess_state, reward, correct = wordle_step(self.alphabet_state, self.guess_state, guess_word, target_word)
            guess_mask = self.guess_mask | F.one_hot(idx, num_classes=vocab_size).bool()  # shape: [1, vocab_size]
            child = MCTSNode(
                alphabet_state=new_alphabet_state,
                guess_state=new_guess_state,
                reward=reward,
                guess_num=self.guess_num + 1,
                guess_mask=guess_mask,
                guess_word=guess_word,
                target_word=target_word,
                parent=self,
                prior=action_prior,
            )
            self.children[idx] = child

    def evaluate(self, actor_critic_net, gamma=0.99):
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
    best_score = -float("inf")
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
    root_alphabet_states,
    root_guess_states,
    guess_num,
    guess_mask_batch,
    target_words,
    actor_critic_net,
    vocab,
    alpha,
    temperature,
    argmax,
    num_simulations=3,
    top_k=3,
    c_puct=1.0,
):
    """
    Run MCTS starting from the given root state for a batch of
    environments and pick the best actions.

    Args:
        root_alphabet_state: shape [batch_size, 26, 11] for Wordle's letter info at the root.
        root_guess_state:    shape [batch_size, max_guesses] for which guess number is active.
        guess_num:           Int, how many guesses used so far.
        guess_mask_batch:    shape [batch_size, max_guesses, action_size] for which words are guessed.
        target_words:        String, the correct Wordle solution.
        actor_critic_net:    Model returning (logits, value).
        vocab:               List of possible guess words (actions).
        alpha, temperature:  Exploration parameters.
        argmax:              Whether to choose the best action or sample.
        num_simulations:     How many times to run the MCTS loop.
        top_k:               Expand top-K children at each node (helps limit branching).
        c_puct:              Exploration constant for the UCB formula.

    Returns:
        batch_action_idx: shape [batch_size] index into 'vocab' of the best action from the root.
        batch_action_visits: shape [batch_size, action_size] number of visits for the chosen action.
    """
    batch_size = root_alphabet_states.size(0)
    action_size = len(vocab)

    batch_action_idx = []
    batch_action_visits = torch.zeros(batch_size, action_size)
    guess_words = ["zzzzz"] * batch_size  # placeholder since the root nodes don't have guesses yet (guess words are used to determine if the node is terminal)

    for i in range(batch_size):
        # 1) Create the root node
        root_node = MCTSNode(
            alphabet_state=root_alphabet_states[i].unsqueeze(0),
            guess_state=root_guess_states[i].unsqueeze(0),
            guess_num=guess_num,
            guess_mask=guess_mask_batch[i].unsqueeze(0),
            guess_word=[guess_words[i]],
            target_word=[target_words[i]],
            parent=None,
            prior=1.0,
            reward=0.0,
        )
        # Expand it once to get initial children
        root_node.expand(actor_critic_net, vocab, alpha, temperature, top_k=top_k)

        # 2) Run repeated simulations
        for sim_num in range(num_simulations):
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
        action_visits = torch.zeros(action_size)
        sum_visits = 0
        best_child, best_action_idx, best_visits = None, None, -1
        for action_idx, child in root_node.children.items():
            action_visits[action_idx] = child.visit_count
            sum_visits += child.visit_count
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_child = child
        batch_action_visits[i] = action_visits

    return batch_action_visits


def select_actions_mcts(actor_critic_net, alphabet_states, guess_states, guess_num, guess_mask_batch, target_words, vocab, alpha, temperature, argmax=False, num_simulations=3, top_k=3, c_puct=1.0):
    """
    Select actions using MCTS for a batch of environments.
    - alphabet_states: [batch_size, 26, 11]
    - guess_states: [batch_size, max_guesses]
    - guess_num: [batch_size] (current guess number)
    - guess_mask_batch: [batch_size, max_guesses, action_size] (mask of already guessed words)
    - target_words: list of length batch_size (each a 5-char string)
    - vocab: list of possible guess words
    - alpha, temperature: float (exploration parameters)
    - num_simulations: int (number of MCTS simulations)
    - top_k: int (number of top children to expand)
    - c_puct: float (exploration constant)
    """
    # Forward pass to get logits and value
    states = torch.cat([alphabet_states.view(-1, 26 * 11), guess_states], dim=-1)  # shape: [batch_size, 26*11 + max_guesses]
    logits, values = actor_critic_net(states)  # shape: logits=[batch_size, action_size], values=[batch_size, 1]
    policy_probs = make_probs(logits, alpha, temperature)  # shape: [batch_size, action_size]

    # # Mask out already guessed words
    # if guess_mask_batch is not None:
    #     guessed = torch.any(guess_mask_batch, dim=1)  # shape: [batch_size, action_size]
    #     probs = probs * ~guessed  # Mask out already guessed words
    #     probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize

    # Get the actions from MCTS
    guess_visits = mcts_search(
        root_alphabet_states=alphabet_states,
        root_guess_states=guess_states,
        guess_num=guess_num,
        guess_mask_batch=guess_mask_batch,
        target_words=target_words,
        actor_critic_net=actor_critic_net,
        vocab=vocab,
        alpha=alpha,
        temperature=temperature,
        argmax=argmax,
        num_simulations=num_simulations,
        top_k=top_k,
        c_puct=c_puct,
    )  # shape: [batch_size], [batch_size, action_size]
    mcts_probs = make_probs(torch.log(guess_visits + 1e-8), 0.0, temperature)  # shape: [batch_size, action_size]  # Use no uniform randomness in MCTS

    # Sample actions
    if not argmax:
        guess_idx = torch.multinomial(mcts_probs, 1).squeeze()  # shape: [batch_size]
    else:
        guess_idx = torch.argmax(mcts_probs, dim=1).squeeze()  # shape: [batch_size]
    guess_words = [vocab[idx] for idx in guess_idx]  # shape: [batch_size]
    guess_idx = F.one_hot(guess_idx, num_classes=mcts_probs.shape[-1]).float()  # shape: [batch_size, action_size]

    return policy_probs, mcts_probs, values, guess_idx, guess_words
