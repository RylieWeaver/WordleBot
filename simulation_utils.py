import torch
import torch.nn.functional as F
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
    words_tensor = torch.tensor(mapped, dtype=torch.long)
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
            torch.tensor([0.2]),  # Number of known occurences
            torch.tensor([0.5]).repeat(5),  # Matching letter and placement (greens)
            torch.tensor([0.02]).repeat(5),  # Unmatching letter and placement (yellows / greys)
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
    rewards += 10.0 * correct_mask  # shape [batch_size]

    return rewards


############################################
# SIMULATION UTILS
############################################
def make_probs(logits, alpha, temperature):
    """
    Returns the final probabilities after mixing with uniform distribution and temperature scaling.
    """
    # Softmax with temperature
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Uniform distribution for alpha-mixing
    uniform_probs = torch.ones_like(probs) / probs.size(-1)
    final_probs = alpha * uniform_probs + (1 - alpha) * probs

    return final_probs


def select_actions(
    actor_critic_net,
    states,
    vocab,
    guess_mask_batch,
    alpha,
    temperature,
    argmax=False,
):
    """
    For each environment in the batch, compute probabilities, and select a word.
    - states: [batch_size, state_size]
    - guess_mask_batch: [batch_size, max_guesses, action_size]
    """
    # Forward pass to get logits and value
    logits, values = actor_critic_net(states)  # shape: logits=[batch_size, action_size], value=[batch_size,1]
    probs = make_probs(logits, alpha, temperature)  # shape: [batch_size, action_size]
    guessed = torch.any(guess_mask_batch, dim=1)  # shape: [batch_size, action_size]
    probs = probs * ~guessed  # Mask out already guessed words

    # Sample an action
    if not argmax:
        guess_idx = torch.multinomial(probs, 1).squeeze()  # shape: [batch_size]
    else:
        guess_idx = torch.argmax(probs, dim=1).squeeze()  # shape: [batch_size]
    guess_words = [vocab[idx] for idx in guess_idx]  # shape: [batch_size]
    guess_idx = F.one_hot(guess_idx, num_classes=probs.shape[-1]).float()  # shape: [batch_size, action_size]

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
    correct = torch.tensor(
        [guess_words[i] == target_words[i] for i in range(batch_size)],
        dtype=torch.bool,
    )

    return new_alphabet_states, new_guess_states, rewards, correct


def collect_episodes(
    actor_critic_net,
    vocab,
    target_words,
    alpha,
    temperature,
    max_guesses=6,
    argmax=False,
):
    """
    Collect a single Wordle episode with a *fixed length* rollout of 6 guesses.
    This will be masked to the actual number of guesses taken.

    Returns:
      states:   Tensor of shape [batch_size, max_guesses+1, state_size]
      log_probs: Tensor of shape [batch_size, max_guesses]
      rewards:  Tensor of shape [batch_size, max_guesses]
      done_mask: Tensor of shape [batch_size, max_guesses+1] (1 = valid step, 0 = after done)
    """
    # 1.0) Setup
    batch_size = len(target_words)
    action_size = len(vocab)

    # 1.1) Initialize Tensors
    states_batch = torch.zeros(
        (batch_size, max_guesses + 1, 26 * 11 + max_guesses),
        dtype=torch.float32,
    )
    probs_batch = torch.zeros((batch_size, max_guesses, action_size), dtype=torch.float32)
    rewards_batch = torch.zeros((batch_size, max_guesses), dtype=torch.float32)
    guess_mask_batch = torch.zeros((batch_size, max_guesses, action_size), dtype=torch.bool)
    active_mask_batch = torch.ones((batch_size, max_guesses + 1), dtype=torch.bool)
    guess_words_batch = []

    # 1.2) Initialize environment state
    alphabet_states = torch.zeros((batch_size, 26, 11), dtype=torch.float32)
    guess_states = torch.zeros([batch_size, max_guesses], dtype=torch.float32)
    guess_states[:, 0] = 1.0

    # 2) Roll out up to max_guesses
    active = torch.ones(batch_size, dtype=torch.bool)
    for t in range(max_guesses):
        # Construct the state
        states = torch.cat([alphabet_states.view(batch_size, -1), guess_states], dim=-1)  # shape [batch_size, 26*11 + max_guesses]
        states_batch[:, t, :] = states

        # Select action
        probs, values, guess_idx, guess_words = select_actions(
            actor_critic_net,
            states,
            vocab,
            guess_mask_batch,
            alpha,
            temperature,
            argmax,
        )

        # Step environment
        alphabet_states, guess_states, rewards, correct = wordle_step(alphabet_states, guess_states, guess_words, target_words)
        active = active & (~correct)

        # Update
        probs_batch[:, t, :] = probs
        rewards_batch[:, t] = rewards
        guess_mask_batch[:, t] = guess_idx
        guess_words_batch.append(guess_words)
        active_mask_batch[:, t + 1] = active

    active_mask_batch[:, -1] = 0  # Last step is not active

    return (
        states_batch,
        probs_batch,
        rewards_batch,
        guess_mask_batch,
        guess_words_batch,
        active_mask_batch,
    )


def process_episodes(
    actor_critic_net,
    states_batch,
    rewards_batch,
    active_mask_batch,
    alpha,
    temperature,
    gamma,
    lam,
):
    """
    Process the collected episodes to compute advantages and final probabilities.

    Takes:
      states_batch:  [batch_size, max_guesses+1, state_size]
      rewards_batch: [batch_size, max_guesses]
      active_mask_batch: [batch_size, max_guesses+1]

    Returns:
      advantages_batch: [batch_size, max_guesses]
      probs_batch: [batch_size, max_guesses, action_size
    """
    batch_size = states_batch.size(0)  # batch size
    max_guesses = states_batch.size(1) - 1  # max guesses (6 for Wordle)

    naive_logits, naive_values_batch = actor_critic_net(
        states_batch
    )  # shape: logits=[batch_size, max_guesses+1, action_size], value=[batch_size, max_guesses+1, 1]
    logits_batch = naive_logits[:, :-1, :]  # shape: [batch_size, max_guesses, action_size]
    values_batch = naive_values_batch.squeeze() * active_mask_batch.float()  # shape: [batch_size, max_guesses+1]
    rewards_batch = rewards_batch * active_mask_batch[:, :-1].float()  # shape: [batch_size, max_guesses]

    # Compute advantages for batch with uniform episode size
    advantages_batch = torch.zeros((batch_size, max_guesses), dtype=torch.float32)
    gae = torch.zeros(batch_size, dtype=torch.float32)
    for t in reversed(range(max_guesses)):
        delta = rewards_batch[:, t] + gamma * values_batch[:, t + 1] - values_batch[:, t]  # Masking values and rewards was important for the GAE here
        gae = delta + gamma * lam * gae
        advantages_batch[:, t] = gae

    # Calculate the final probs
    probs_batch = make_probs(logits_batch, alpha, temperature)

    return advantages_batch, probs_batch
