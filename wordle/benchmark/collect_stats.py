# General
import os
from tqdm import tqdm

# Torch

# Wordle



def get_stats_from_dir(stats_dir):
    # Setup
    total_correct = 0
    total_guesses = 0

    # Collect all the files inside the stats_dir
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
        print(f"Created directory: {stats_dir}")

    # Aggregate stats from existing "word_*" files if present
    stats_files = [f for f in os.listdir(stats_dir) if f.startswith("word_") and f.endswith(".txt")]
    for i in tqdm(range(len(stats_files)), desc="Aggregating stats"):
        fname = stats_files[i]
        with open(os.path.join(stats_dir, fname), "r") as f:
            line = f.readline()
            # Example line: Target: aback, Correct: True, Guesses: 4
            parts = line.strip().split(",")
            correct_str = parts[1].split(":")[1].strip()
            guesses_str = parts[2].split(":")[1].strip()
            correct = 1 if correct_str == "True" else 0
            guesses = int(guesses_str)
            total_correct += correct
            total_guesses += guesses

    # Average stats
    avg_correct = float(total_correct) / float(len(stats_files))
    avg_guesses = float(total_guesses) / float(len(stats_files))

    # Save the average stats to a file
    with open(os.path.join(stats_dir, "average_stats.txt"), "w") as f:
        f.write(f"Average correct: {avg_correct}\n")
        f.write(f"Average guesses: {avg_guesses}\n")

    return avg_correct, avg_guesses


if __name__ == "__main__":
    # Setup
    stats_dir = "/Users/rylie/Coding/Projects/Wordle_RL/wordle/benchmark/minmax_stats/"

    # Get stats
    avg_correct, avg_guesses = get_stats_from_dir(stats_dir)
    print(f"Average correct: {avg_correct}")
    print(f"Average guesses: {avg_guesses}")
