#################################################
# STORAGE TO GO BACK TO OTHER VERSIONS IF NEEDED
#################################################

import torch


############################################
# STATE UTILS
############################################
def update_letter_state(
    given_letter_state,
    guess_letter_state,
    state_green_idx,
    guess_green_idx,
    state_yellow_idx,
    guess_yellow_idx,
):
    """
    Extracted verbatim from the original code's merging logic.
    Creates and returns new_letter_state [12].
    """
    new_letter_state = torch.zeros(12, dtype=torch.float32)

    # Logic if there are greens in both
    if guess_green_idx and state_green_idx:
        ## If they are the same greens, we can combine the yellow information
        if guess_green_idx == state_green_idx:
            new_letter_state[7:12] = guess_letter_state[7:12] = given_letter_state[
                7:12
            ]  # Equal green information
            new_letter_state[2:7] = torch.max(
                guess_letter_state[2:7], given_letter_state[2:7]
            )  # Combined the yellow info
            new_letter_state[1] = 0  # Explicitly say it is not grey
            new_letter_state[0] = 0  # Explicitly say it is not unknown
        if guess_green_idx != state_green_idx:
            new_letter_state[7:12] = torch.max(
                guess_letter_state[7:12], given_letter_state[7:12]
            )  # Combined green info
            new_letter_state[2:7] = torch.zeros(
                5, dtype=torch.float32
            )  # No yellow information
            new_letter_state[1] = 0  # Explicitly say it is not grey
            new_letter_state[0] = 0  # Explicitly say it is not unknown

    # Logic if there is a green in one and not the other
    if guess_green_idx and not state_green_idx:
        new_letter_state[7:12] = guess_letter_state[7:12]  # Green information
        new_letter_state[2:7] = torch.zeros(
            5, dtype=torch.float32
        )  # No yellow information
        new_letter_state[1] = 0  # Explicitly say it is not grey
        new_letter_state[0] = 0  # Explicitly say it is not unknown
    if state_green_idx and not guess_green_idx:
        new_letter_state[7:12] = given_letter_state[7:12]
        new_letter_state[2:7] = torch.zeros(5, dtype=torch.float32)  # No yellow
        new_letter_state[1] = 0  # Explicitly say it is not grey
        new_letter_state[0] = 0  # Explicitly say it is not unknown

    # Logic if there are no greens
    if not guess_green_idx and not state_green_idx:
        if guess_yellow_idx or state_yellow_idx:
            new_letter_state[7:12] = torch.zeros(
                5, dtype=torch.float32
            )  # No green information
            new_letter_state[2:7] = torch.max(
                guess_letter_state[2:7], given_letter_state[2:7]
            )
            # Combining the yellow information
            new_letter_state[1] = 0  # Explicitly say it is not grey
            new_letter_state[0] = 0  # Explicitly say it is not unknown
        # Logic if no yellows as well
        if not guess_yellow_idx and not state_yellow_idx:
            new_letter_state[7:12] = torch.zeros(
                5, dtype=torch.float32
            )  # No green info
            new_letter_state[2:7] = torch.zeros(
                5, dtype=torch.float32
            )  # No yellow info
            new_letter_state[1] = 1  # If no yellows or greens, must be grey
            new_letter_state[0] = 0  # Explicitly say it is not unknown

    return new_letter_state


def update_alphabet_state(given_alphabet_state, guess_word, target_word):
    """
    Simulate one Wordle step (single sample):
      - given_alphabet_state: [26, 12] tensor
      - guess_word: string of length 5
      - target_word: string of length 5

    Returns:
      new_alphabet_state: [26, 12] tensor
      reward: float (based on discovered info + correct guess)
      done: bool
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    new_alphabet_state = torch.zeros_like(given_alphabet_state)

    for i, (letter, old_letter_state) in enumerate(zip(alphabet, given_alphabet_state)):
        guess_letter_state = torch.zeros(12, dtype=torch.float32)
        guess_letter_idx = [idx for idx, ch in enumerate(guess_word) if ch == letter]
        if not guess_letter_idx:
            # Letter not in guess => copy old state
            new_alphabet_state[i] = old_letter_state
            continue

        # Build guess_letter_state for each position of this letter
        for idx in guess_letter_idx:
            if guess_word[idx] == target_word[idx]:
                guess_letter_state[7 + idx] = 1  # Green
            elif guess_word[idx] in target_word:
                guess_letter_state[2 + idx] = 1  # Yellow
            else:
                guess_letter_state[1] = 1  # Grey

        # Indices from the old letter state
        state_green_idx = [
            j + 7 for j, val in enumerate(old_letter_state[7:12]) if val == 1
        ]
        state_yellow_idx = [
            j + 2 for j, val in enumerate(old_letter_state[2:7]) if val == 1
        ]
        # Indices from the guess
        guess_green_idx = [
            j + 7 for j, val in enumerate(guess_letter_state[7:12]) if val == 1
        ]
        guess_yellow_idx = [
            j + 2 for j, val in enumerate(guess_letter_state[2:7]) if val == 1
        ]

        # Update letter state
        new_letter_state = update_letter_state(
            old_letter_state,
            guess_letter_state,
            state_green_idx,
            guess_green_idx,
            state_yellow_idx,
            guess_yellow_idx,
        )
        new_alphabet_state[i] = new_letter_state

    return new_alphabet_state
