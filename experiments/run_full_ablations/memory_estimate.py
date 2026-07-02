# General
import argparse
import csv
import json
import math
import sys


BYTES_PER_GB = 1024 ** 3
FP_BYTES = 4
BOOL_BYTES = 1
INT64_BYTES = 8

DEFAULT_VOCAB_SIZE = 12972
DEFAULT_TARGET_VOCAB_SIZE = 2315

RUN_DEFAULTS = {
    "model-name": "DotGuessStateNet",
    "model-size-multiplier": 2.0,
    "layers": 3,
    "m": 3,
    "loader-batch-size": 32,
    "processing-batch-size": 16,
    "processing-num-workers": 4,
    "repeats": 16,
    "num-workers": 4,
    "max-guesses": 6,
}


def merged_run_args(ablation_args, default_args=None, run_defaults=None):
    merged = dict(run_defaults or RUN_DEFAULTS)
    if default_args:
        merged.update(default_args)
    merged.update(ablation_args)
    return merged


def _gb(num_bytes):
    return num_bytes / BYTES_PER_GB


def _ceil_gb(gb):
    return int(math.ceil(gb))


def _int_arg(run_args, key):
    return int(run_args.get(key, RUN_DEFAULTS[key]))


def _float_arg(run_args, key):
    return float(run_args.get(key, RUN_DEFAULTS[key]))


def _model_name(run_args):
    return str(run_args.get("model-name", RUN_DEFAULTS["model-name"]))


def _linear_params(input_dim, output_dim, bias=True):
    return input_dim * output_dim + (output_dim if bias else 0)


def _layernorm_params(dim):
    return 2 * dim


def _mlp_block_params(dim):
    return _layernorm_params(dim) + _linear_params(dim, dim)


def _ff_params(dim):
    return _linear_params(dim, 4 * dim) + _linear_params(4 * dim, dim)


def _mha_params(dim, bias=False):
    return _linear_params(dim, 3 * dim, bias=bias)


def _transformer_block_params(dim, bias=False):
    return (
        _layernorm_params(dim) +
        _mha_params(dim, bias=bias) +
        _layernorm_params(dim) +
        _ff_params(dim)
    )


def _model_dims(run_args, vocab_size):
    c = _float_arg(run_args, "model-size-multiplier")
    G = _int_arg(run_args, "max-guesses")
    return {
        "V": int(vocab_size),
        "G": G,
        "IS": (26 * 11) + G,
        "IG": 26 * 5,
        "H": int(128 * c),
        "HS": int(128 * c),
        "HG": int(8 * c),
        "O": int(128 * c),
        "layers": _int_arg(run_args, "layers"),
    }


def estimate_model_params(run_args, vocab_size):
    dims = _model_dims(run_args, vocab_size)
    name = _model_name(run_args)
    L = dims["layers"]

    if name == "ActorCriticNet":
        I, H, V = dims["IS"], dims["H"], dims["V"]
        return (
            _linear_params(I, H) + _layernorm_params(H) +
            L * _mlp_block_params(H) +
            _mlp_block_params(H) + _layernorm_params(H) + _linear_params(H, V) +
            _mlp_block_params(H) + _layernorm_params(H) + _linear_params(H, 1)
        )

    if name == "DotGuessStateNet":
        IS, IG, HS, HG, O = dims["IS"], dims["IG"], dims["HS"], dims["HG"], dims["O"]
        return (
            _layernorm_params(IS) + _linear_params(IS, HS) +
            _layernorm_params(IG) + _linear_params(IG, HG) +
            L * (_mlp_block_params(HS) + _mlp_block_params(HG)) +
            _mlp_block_params(HS) + _layernorm_params(HS) + _linear_params(HS, O, bias=False) +
            _mlp_block_params(HG) + _layernorm_params(HG) + _linear_params(HG, O, bias=False) +
            _linear_params(HS + IG, HS) + _mlp_block_params(HS) +
            _layernorm_params(HS) + _linear_params(HS, 1) +
            1
        )

    if name == "WordleTransformer":
        IS, IG, HS, HG, O = dims["IS"], dims["IG"], dims["HS"], dims["HG"], dims["O"]
        return (
            _linear_params(IS, HS) + _layernorm_params(HS) +
            _linear_params(IG, HG) + _layernorm_params(HG) +
            L * (_transformer_block_params(HS) + _transformer_block_params(HG)) +
            _mlp_block_params(HS) + _layernorm_params(HS) +
            _linear_params(HS, O) + _layernorm_params(O) +
            _mlp_block_params(HG) + _layernorm_params(HG) +
            _linear_params(HG, O) + _layernorm_params(O) +
            _mlp_block_params(HS) + _layernorm_params(HS) + _linear_params(HS, 1)
        )

    raise ValueError(f"Unknown model name: {name}")


def estimate_model_buffer_bytes(run_args, vocab_size):
    name = _model_name(run_args)
    if name == "ActorCriticNet":
        return 0
    dims = _model_dims(run_args, vocab_size)
    V, IG, O = dims["V"], dims["IG"], dims["O"]
    return (
        V * 5 * INT64_BYTES +     # total_vocab_tensor
        V * IG * FP_BYTES +       # guess_states
        V * O * FP_BYTES          # guess_k_static
    )


def estimate_host_episode_bytes(run_args, vocab_size, target_vocab_size):
    N = int(target_vocab_size)
    V = int(vocab_size)
    T = int(target_vocab_size)
    R = _int_arg(run_args, "repeats")
    G = _int_arg(run_args, "max-guesses")
    rows = N * R

    state_step = (
        INT64_BYTES +                 # t
        BOOL_BYTES +                  # last_guess
        26 * 11 * FP_BYTES +          # alphabet
        BOOL_BYTES +                  # active_mask
        V * BOOL_BYTES +              # guessed_mask
        T * BOOL_BYTES +              # target_mask
        FP_BYTES +                    # entropy
        V * BOOL_BYTES +              # total_target_mask
        INT64_BYTES                   # idx
    )
    action_step = (
        4 * V * FP_BYTES +            # policy/masked/mixed/mixed_masked probs
        INT64_BYTES +                 # guess_idx
        V * BOOL_BYTES +              # guess_mask
        V * BOOL_BYTES                # valid_mask
    )
    response = rows * (
        (G + 1) * FP_BYTES +          # values
        G * 4 * FP_BYTES +            # rewards, expected rewards/values, correct
        G * 3 * FP_BYTES              # returns, advantages, norm_advantages
    )
    loader_vocab = (
        T * 5 * INT64_BYTES +
        V * 5 * INT64_BYTES +
        T * 26 * 11 * FP_BYTES +
        V * 26 * 11 * FP_BYTES
    )
    dataloader_workers = max(_int_arg(run_args, "num-workers"), 0)
    processing_workers = max(_int_arg(run_args, "processing-num-workers"), 0)
    python_overhead = 2.0 * BYTES_PER_GB

    components = {
        "host_states_gb": _gb((G + 1) * rows * state_step),
        "host_actions_gb": _gb(G * rows * action_step),
        "host_responses_gb": _gb(response),
        "host_loader_vocab_gb": _gb(loader_vocab),
        "host_worker_overhead_gb": (dataloader_workers + processing_workers) * 0.5,
        "host_python_overhead_gb": _gb(python_overhead),
    }
    components["host_raw_gb"] = sum(components.values())
    return components


def estimate_gpu_model_bytes(run_args, vocab_size):
    params = estimate_model_params(run_args, vocab_size)
    buffers = estimate_model_buffer_bytes(run_args, vocab_size)
    model = params * FP_BYTES + buffers
    optimizer = params * FP_BYTES * 3       # grad, AdamW exp_avg, AdamW exp_avg_sq
    return (3 * model) + optimizer


def estimate_gpu_activation_bytes(run_args, vocab_size, num_states):
    dims = _model_dims(run_args, vocab_size)
    name = _model_name(run_args)
    V = int(vocab_size)
    L = dims["layers"]

    if name == "ActorCriticNet":
        H = dims["H"]
        hidden = num_states * H * (L + 6) * FP_BYTES
        logits = num_states * V * FP_BYTES
        return 3 * (hidden + logits)

    if name == "DotGuessStateNet":
        HS, HG, O = dims["HS"], dims["HG"], dims["O"]
        state_hidden = num_states * HS * (L + 6) * FP_BYTES
        query_and_logits = num_states * (O + V) * FP_BYTES
        guess_cache_graph = V * HG * (L + 4) * FP_BYTES + V * O * FP_BYTES
        return 3 * (state_hidden + query_and_logits + guess_cache_graph)

    if name == "WordleTransformer":
        HS, HG, O = dims["HS"], dims["HG"], dims["O"]
        state_hidden = num_states * HS * (2 * L + 6) * FP_BYTES
        query_and_logits = num_states * (O + V) * FP_BYTES
        guess_cache_graph = V * HG * (2 * L + 4) * FP_BYTES + V * O * FP_BYTES
        return 3 * (state_hidden + query_and_logits + guess_cache_graph)

    raise ValueError(f"Unknown model name: {name}")


def estimate_dot_product_bytes(run_args, vocab_size, num_states, training):
    dims = _model_dims(run_args, vocab_size)
    name = _model_name(run_args)
    V = int(vocab_size)

    if name == "ActorCriticNet":
        return 0

    if name in {"DotGuessStateNet", "WordleTransformer"}:
        O = dims["O"]
        q = num_states * O * FP_BYTES
        k = V * O * FP_BYTES
        scores_scratch = num_states * V * FP_BYTES
        backward_scratch = scores_scratch if training else 0
        return q + k + scores_scratch + backward_scratch

    raise ValueError(f"Unknown model name: {name}")


def estimate_episode_batch_bytes(run_args, vocab_size, target_vocab_size, batch_size):
    V = int(vocab_size)
    T = int(target_vocab_size)
    R = _int_arg(run_args, "repeats")
    G = _int_arg(run_args, "max-guesses")
    rows = int(batch_size) * R

    state_step = (
        INT64_BYTES +
        BOOL_BYTES +
        26 * 11 * FP_BYTES +
        BOOL_BYTES +
        V * BOOL_BYTES +
        T * BOOL_BYTES +
        FP_BYTES +
        V * BOOL_BYTES +
        INT64_BYTES
    )
    action_step = (
        4 * V * FP_BYTES +
        INT64_BYTES +
        V * BOOL_BYTES +
        V * BOOL_BYTES
    )
    response = rows * (
        (G + 1) * FP_BYTES +
        G * 4 * FP_BYTES +
        G * 3 * FP_BYTES
    )
    return ((G + 1) * rows * state_step) + (G * rows * action_step) + response


def estimate_gpu_train_bytes(run_args, vocab_size, target_vocab_size):
    P = _int_arg(run_args, "processing-batch-size")
    R = _int_arg(run_args, "repeats")
    G = _int_arg(run_args, "max-guesses")
    V = int(vocab_size)
    states = (G + 1) * P * R
    turns = G * P * R

    episode_batch = estimate_episode_batch_bytes(run_args, vocab_size, target_vocab_size, P)
    current_probs = 4 * turns * V * FP_BYTES
    ref_best_probs = 2 * 4 * states * V * FP_BYTES
    loss_workspace = 12 * turns * V * FP_BYTES
    model = estimate_gpu_model_bytes(run_args, vocab_size)
    activations = estimate_gpu_activation_bytes(run_args, vocab_size, states)
    dot_product = estimate_dot_product_bytes(run_args, vocab_size, states, training=True)
    allocator = 2.0 * BYTES_PER_GB

    components = {
        "gpu_train_episode_batch_gb": _gb(episode_batch),
        "gpu_train_current_probs_gb": _gb(current_probs),
        "gpu_train_ref_best_probs_gb": _gb(ref_best_probs),
        "gpu_train_loss_workspace_gb": _gb(loss_workspace),
        "gpu_train_model_optimizer_gb": _gb(model),
        "gpu_train_activations_gb": _gb(activations),
        "gpu_train_dot_product_gb": _gb(dot_product),
        "gpu_train_allocator_gb": _gb(allocator),
    }
    components["gpu_train_raw_gb"] = sum(components.values())
    return components


def estimate_gpu_collect_bytes(run_args, vocab_size, target_vocab_size):
    B = _int_arg(run_args, "loader-batch-size")
    R = _int_arg(run_args, "repeats")
    m = _int_arg(run_args, "m")
    V = int(vocab_size)
    T = int(target_vocab_size)
    rows = B * R

    model = estimate_gpu_model_bytes(run_args, vocab_size)
    dot_product = estimate_dot_product_bytes(run_args, vocab_size, rows * (1 + m), training=False)
    step_states = rows * (
        26 * 11 * FP_BYTES +
        2 * V * BOOL_BYTES +
        T * BOOL_BYTES
    )
    sample_probs = rows * V * (5 * FP_BYTES + 2 * BOOL_BYTES)
    candidate_logits = rows * m * V * FP_BYTES
    candidate_masks = rows * m * (V + T) * BOOL_BYTES
    entropy_compare = rows * (1 + m) * T * 26 * 11 * BOOL_BYTES
    vocab_data = (
        T * 26 * 11 * FP_BYTES +
        V * 26 * 11 * FP_BYTES +
        (T + V) * 5 * INT64_BYTES
    )
    allocator = 2.0 * BYTES_PER_GB

    components = {
        "gpu_collect_model_gb": _gb(model),
        "gpu_collect_dot_product_gb": _gb(dot_product),
        "gpu_collect_step_states_gb": _gb(step_states),
        "gpu_collect_sample_probs_gb": _gb(sample_probs),
        "gpu_collect_candidate_logits_gb": _gb(candidate_logits),
        "gpu_collect_candidate_masks_gb": _gb(candidate_masks),
        "gpu_collect_entropy_compare_gb": _gb(entropy_compare),
        "gpu_collect_vocab_data_gb": _gb(vocab_data),
        "gpu_collect_allocator_gb": _gb(allocator),
    }
    components["gpu_collect_raw_gb"] = sum(components.values())
    return components


def estimate_memory_components(run_args, vocab_size, target_vocab_size, buffer, min_gb):
    host = estimate_host_episode_bytes(run_args, vocab_size, target_vocab_size)
    gpu_train = estimate_gpu_train_bytes(run_args, vocab_size, target_vocab_size)
    gpu_collect = estimate_gpu_collect_bytes(run_args, vocab_size, target_vocab_size)
    host_buffered = host["host_raw_gb"] * buffer
    gpu_train_buffered = gpu_train["gpu_train_raw_gb"] * buffer
    gpu_collect_buffered = gpu_collect["gpu_collect_raw_gb"] * buffer
    gpu_raw = max(gpu_train["gpu_train_raw_gb"], gpu_collect["gpu_collect_raw_gb"])
    gpu_buffered = max(gpu_train_buffered, gpu_collect_buffered)
    cpu_mem_gb = max(min_gb, _ceil_gb(host_buffered))
    gpu_mem_gb = _ceil_gb(gpu_buffered)
    return {
        **host,
        **gpu_train,
        **gpu_collect,
        "cpu_raw_gb": host["host_raw_gb"],
        "cpu_buffered_gb": host_buffered,
        "cpu_mem_gb": cpu_mem_gb,
        "cpu_mem": f"{cpu_mem_gb}G",
        "gpu_raw_gb": gpu_raw,
        "gpu_buffered_gb": gpu_buffered,
        "gpu_mem_gb": gpu_mem_gb,
        "gpu_mem": f"{gpu_mem_gb}G",
        "host_buffered_gb": host_buffered,
        "gpu_train_buffered_gb": gpu_train_buffered,
        "gpu_collect_buffered_gb": gpu_collect_buffered,
        "slurm_mem_gb": cpu_mem_gb,
        "slurm_mem": f"{cpu_mem_gb}G",
    }


def estimate_slurm_mem(run_args, vocab_size, buffer, min_gb, target_vocab_size=DEFAULT_TARGET_VOCAB_SIZE):
    components = estimate_memory_components(
        run_args,
        vocab_size=vocab_size,
        target_vocab_size=target_vocab_size,
        buffer=buffer,
        min_gb=min_gb,
    )
    return components["slurm_mem_gb"]


def estimate_slurm_mem_str(run_args, vocab_size, buffer, min_gb, target_vocab_size=DEFAULT_TARGET_VOCAB_SIZE):
    return f"{estimate_slurm_mem(run_args, vocab_size, buffer, min_gb, target_vocab_size)}G"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true")
    output.add_argument("--csv", action="store_true")
    parser.add_argument("--runs-per-ablation", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default=RUN_DEFAULTS["model-name"])
    parser.add_argument("--model-size-multiplier", type=float, default=RUN_DEFAULTS["model-size-multiplier"])
    parser.add_argument("--layers", type=int, default=RUN_DEFAULTS["layers"])
    parser.add_argument("--m", type=int, default=RUN_DEFAULTS["m"])
    parser.add_argument("--loader-batch-size", type=int, default=RUN_DEFAULTS["loader-batch-size"])
    parser.add_argument("--processing-batch-size", type=int, default=RUN_DEFAULTS["processing-batch-size"])
    parser.add_argument("--processing-num-workers", type=int, default=RUN_DEFAULTS["processing-num-workers"])
    parser.add_argument("--repeats", type=int, default=RUN_DEFAULTS["repeats"])
    parser.add_argument("--num-workers", type=int, default=RUN_DEFAULTS["num-workers"])
    parser.add_argument("--max-guesses", type=int, default=RUN_DEFAULTS["max-guesses"])
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--target-vocab-size", type=int, default=DEFAULT_TARGET_VOCAB_SIZE)
    parser.add_argument("--buffer", type=float, default=1.5)
    parser.add_argument("--min-gb", type=int, default=0)
    return parser


def row_for_args(name, ablation_idx, run_args, args):
    components = estimate_memory_components(
        run_args,
        vocab_size=args.vocab_size,
        target_vocab_size=args.target_vocab_size,
        buffer=args.buffer,
        min_gb=args.min_gb,
    )
    return {
        "ablation": name,
        "name": name,
        "ablation_idx": ablation_idx,
        "mem": components["slurm_mem"],
        "mem_gb": components["slurm_mem_gb"],
        "cpu_mem": components["cpu_mem"],
        "cpu_mem_gb": components["cpu_mem_gb"],
        "gpu_mem": components["gpu_mem"],
        "gpu_mem_gb": components["gpu_mem_gb"],
        "cpu_raw_gb": components["cpu_raw_gb"],
        "cpu_buffered_gb": components["cpu_buffered_gb"],
        "gpu_raw_gb": components["gpu_raw_gb"],
        "gpu_buffered_gb": components["gpu_buffered_gb"],
        "slurm_mem": components["slurm_mem"],
        "slurm_mem_gb": components["slurm_mem_gb"],
        "host_raw_gb": components["host_raw_gb"],
        "host_buffered_gb": components["host_buffered_gb"],
        "gpu_train_raw_gb": components["gpu_train_raw_gb"],
        "gpu_train_buffered_gb": components["gpu_train_buffered_gb"],
        "gpu_collect_raw_gb": components["gpu_collect_raw_gb"],
        "gpu_collect_buffered_gb": components["gpu_collect_buffered_gb"],
        "model_name": _model_name(run_args),
        "model_size_multiplier": _float_arg(run_args, "model-size-multiplier"),
        "layers": _int_arg(run_args, "layers"),
        "m": _int_arg(run_args, "m"),
        "repeats": _int_arg(run_args, "repeats"),
        "loader_batch_size": _int_arg(run_args, "loader-batch-size"),
        "processing_batch_size": _int_arg(run_args, "processing-batch-size"),
        "processing_num_workers": _int_arg(run_args, "processing-num-workers"),
        "max_guesses": _int_arg(run_args, "max-guesses"),
        "vocab_size": args.vocab_size,
        "target_vocab_size": args.target_vocab_size,
        "buffer": args.buffer,
        "min_gb": args.min_gb,
        **components,
    }


def print_table(rows):
    width = max(len(row["name"]) for row in rows) if rows else 4
    print(
        f"{'name'.ljust(width)}  cpu_mem  cpu_raw  gpu_mem  gpu_raw  gpu_train  gpu_collect  "
        "model              c  layers  m  repeats"
    )
    print(
        f"{'-' * width}  -------  -------  -------  -------  ---------  -----------  "
        "-----------------  -  ------  -  -------"
    )
    for row in rows:
        print(
            f"{row['name'].ljust(width)}  "
            f"{row['cpu_mem']:>7}  "
            f"{row['cpu_raw_gb']:>7.1f}  "
            f"{row['gpu_mem']:>7}  "
            f"{row['gpu_raw_gb']:>7.1f}  "
            f"{row['gpu_train_raw_gb']:>9.1f}  "
            f"{row['gpu_collect_raw_gb']:>11.1f}  "
            f"{row['model_name']:<17}  "
            f"{row['model_size_multiplier']:>1g}  "
            f"{row['layers']:>6}  "
            f"{row['m']:>1}  "
            f"{row['repeats']:>7}"
        )


def print_csv(rows):
    fieldnames = [
        "ablation",
        "ablation_idx",
        "run_name",
        "run_idx",
        "seed",
        "mem",
        "mem_gb",
        "cpu_mem",
        "cpu_mem_gb",
        "gpu_mem",
        "gpu_mem_gb",
        "cpu_raw_gb",
        "cpu_buffered_gb",
        "gpu_raw_gb",
        "gpu_buffered_gb",
        "slurm_mem",
        "slurm_mem_gb",
        "host_raw_gb",
        "host_buffered_gb",
        "gpu_train_raw_gb",
        "gpu_train_buffered_gb",
        "gpu_collect_raw_gb",
        "gpu_collect_buffered_gb",
        "model_name",
        "model_size_multiplier",
        "layers",
        "m",
        "repeats",
        "loader_batch_size",
        "processing_batch_size",
        "processing_num_workers",
        "max_guesses",
        "vocab_size",
        "target_vocab_size",
        "buffer",
        "min_gb",
        "host_states_gb",
        "host_actions_gb",
        "host_responses_gb",
        "host_loader_vocab_gb",
        "host_worker_overhead_gb",
        "host_python_overhead_gb",
        "gpu_train_episode_batch_gb",
        "gpu_train_current_probs_gb",
        "gpu_train_ref_best_probs_gb",
        "gpu_train_loss_workspace_gb",
        "gpu_train_model_optimizer_gb",
        "gpu_train_activations_gb",
        "gpu_train_dot_product_gb",
        "gpu_train_allocator_gb",
        "gpu_collect_model_gb",
        "gpu_collect_dot_product_gb",
        "gpu_collect_step_states_gb",
        "gpu_collect_sample_probs_gb",
        "gpu_collect_candidate_logits_gb",
        "gpu_collect_candidate_masks_gb",
        "gpu_collect_entropy_compare_gb",
        "gpu_collect_vocab_data_gb",
        "gpu_collect_allocator_gb",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def sweep_rows(args):
    import run_sweep

    rows = []
    for ablation_idx, ablation in enumerate(run_sweep.ABLATIONS):
        run_args = merged_run_args(ablation["args"], default_args=run_sweep.DEFAULT_ARGS)
        base_row = row_for_args(ablation["name"], ablation_idx, run_args, args)
        if args.csv:
            for run_idx in range(args.runs_per_ablation):
                seed = args.base_seed + 1000 * ablation_idx + run_idx
                rows.append({
                    **base_row,
                    "run_idx": run_idx,
                    "seed": seed,
                    "run_name": f"run_{run_idx:02d}_seed_{seed}",
                })
        else:
            rows.append(base_row)
    return rows


def main():
    args = build_parser().parse_args()
    try:
        if args.sweep:
            rows = sweep_rows(args)
            if args.json:
                print(json.dumps(rows, indent=4))
            elif args.csv:
                print_csv(rows)
            else:
                print_table(rows)
            return

        run_args = {
            "model-name": args.model_name,
            "model-size-multiplier": args.model_size_multiplier,
            "layers": args.layers,
            "m": args.m,
            "loader-batch-size": args.loader_batch_size,
            "processing-batch-size": args.processing_batch_size,
            "processing-num-workers": args.processing_num_workers,
            "repeats": args.repeats,
            "num-workers": args.num_workers,
            "max-guesses": args.max_guesses,
        }
        result = row_for_args("single", 0, run_args, args)
        if args.json:
            print(json.dumps(result, indent=4))
        elif args.csv:
            print_csv([{
                "run_name": "single",
                "run_idx": 0,
                "seed": "",
                **result,
            }])
        else:
            print(f"cpu_mem={result['cpu_mem']} gpu_mem={result['gpu_mem']}")
    except BrokenPipeError:
        return


if __name__ == "__main__":
    main()
