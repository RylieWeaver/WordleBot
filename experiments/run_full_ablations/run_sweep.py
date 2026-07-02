# General
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from memory_estimate import estimate_slurm_mem, merged_run_args


RUN_DIR = Path(__file__).resolve().parent

DEFAULT_ARGS = {
    "model-name": "DotGuessStateNet",
    "model-size-multiplier": 1.0,
    "layers": 3,
    "m": 3,
    "loader-batch-size": 32,
    "processing-batch-size": 16,
    "repeats": 16,
    "num-workers": 4,
    "max-guesses": 6,
    "gamma": 1.0,
    "lam": 1.0,
    "batches-per-gradient-step": 16,
    "rollout-size": 4,
    "correct-blend-factor": 1.0,
    "reward-blend-factor": 1.0,
    "value-blend-factor": 1.0,
    "kl-guide-loss-weight": 0.25,
    "kl-best-loss-weight": 0.10,
    "adv-mean-reduce-dims": "2",
    "adv-std-reduce-dims": "0,2",
    "advantage-type": "gae",
    "use-inductive-biases": True,
    "correct-reward": 0.1,
    "amp-dtype": "bfloat16",
    "learning-rate": 1e-4,
}

def is_default_args(args):
    return all(DEFAULT_ARGS.get(key) == value for key, value in args.items())


def ablation(name, args):
    if is_default_args(args):
        # Don't re-train the default args over and over
        return None
    return {"name": name, "args": args}


def build_ablations():
    ablations = [{"name": "baseline", "args": {}}]

    bias_configs = [
        ("no_bias", {"use-inductive-biases": False}),
        ("bias_kl_guide_0", {"use-inductive-biases": True, "kl-guide-loss-weight": 0.0}),
        ("bias_default_kl_guide", {"use-inductive-biases": True}),
    ]
    for model_name in ["ActorCriticNet", "DotGuessStateNet"]:
        for bias_name, bias_args in bias_configs:
            args = {"model-name": model_name, **bias_args}
            item = ablation(f"model_{model_name.lower()}__{bias_name}", args)
            if item:
                ablations.append(item)

    for c in [1.0, 2.0, 4.0, 8.0]:
        item = ablation(f"model_size_{c:g}", {"model-size-multiplier": c})
        if item:
            ablations.append(item)

    for layers in [1, 2, 3, 4, 6]:
        item = ablation(f"layers_{layers}", {"layers": layers})
        if item:
            ablations.append(item)

    for m in [1, 2, 3, 4, 5]:
        item = ablation(f"m_{m}", {"m": m})
        if item:
            ablations.append(item)

    for gamma in [0.2, 0.5, 1.0]:
        item = ablation(f"gamma_{gamma:g}", {"gamma": gamma})
        if item:
            ablations.append(item)

    for lam in [0.2, 0.5, 1.0]:
        item = ablation(f"lambda_{lam:g}", {"lam": lam})
        if item:
            ablations.append(item)

    ablations.append({"name": "correct_reward_0", "args": {"correct-reward": 0.0}})
    ablations.append({"name": "kl_best_loss_weight_0", "args": {"kl-best-loss-weight": 0.0}})

    # NOTE: Define a contour for equal batch size and episodes per rollout
    repeat_contour = [
        {"repeats": 8, "batches-per-gradient-step": 32, "rollout-size": 8},
        {"repeats": 16, "batches-per-gradient-step": 16, "rollout-size": 4},
        {"repeats": 32, "batches-per-gradient-step": 8, "rollout-size": 2},
        {"repeats": 64, "batches-per-gradient-step": 4, "rollout-size": 1},
    ]
    for args in repeat_contour:
        item = ablation(
            f"repeats_{args['repeats']}__bpg_{args['batches-per-gradient-step']}__rollouts_{args['rollout-size']}",
            args,
        )
        if item:
            ablations.append(item)

    blend_factors = [0.0, 0.1, 0.5, 1.0]
    for key in ["correct-blend-factor", "reward-blend-factor", "value-blend-factor"]:
        for value in blend_factors:
            item = ablation(f"{key.replace('-', '_')}_{value:g}", {key: value})
            if item:
                ablations.append(item)

    reduce_dims = [
        ("group", "2"),
        ("batch_group", "0,2"),
        ("global", "0,1,2"),
    ]
    for mean_name, mean_dims in reduce_dims:
        for std_name, std_dims in reduce_dims:
            args = {
                "adv-mean-reduce-dims": mean_dims,
                "adv-std-reduce-dims": std_dims,
            }
            item = ablation(f"adv_norm_mean_{mean_name}__std_{std_name}", args)
            if item:
                ablations.append(item)

    for advantage_type in [
        "gae",
        "reward-total",
        "reward-telescoped",
        "reward-telescoped-value-baseline",
    ]:
        item = ablation(f"advantage_{advantage_type}", {"advantage-type": advantage_type})
        if item:
            ablations.append(item)

    return ablations


ABLATIONS = build_ablations()


# Slug: Make a string safe for use in file paths and command-line arguments
def slugify(value):
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip())
    return value.strip("_").lower()


def cli_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def add_cli_args(cmd, args):
    for key, value in args.items():
        cmd.extend([f"--{key}", cli_value(value)])


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=4)


def append_sweep_log(path, message):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(message, flush=True)
    with path.open("a") as f:
        f.write(message + "\n")


def shell_join(cmd):
    return " ".join(shlex.quote(str(part)) for part in cmd)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def slurm_mem(args, ablation_args):
    if args.slurm_mem:
        return args.slurm_mem
    run_args = merged_run_args(ablation_args, default_args=DEFAULT_ARGS)
    mem_gb = estimate_slurm_mem(
        run_args,
        vocab_size=args.slurm_vocab_size,
        target_vocab_size=args.slurm_target_vocab_size,
        buffer=args.slurm_mem_buffer,
        min_gb=args.slurm_min_mem_gb,
    )
    return f"{mem_gb}G"


def slurm_directives(args, name, mem):
    directives = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={args.slurm_cpus_per_task}",
        f"#SBATCH --time={args.slurm_time}",
        f"#SBATCH --mem={mem}",
    ]
    if args.slurm_gpu_directive.lower() not in {"", "none"}:
        directives.append(f"#SBATCH {args.slurm_gpu_directive}")
    if args.slurm_mem_per_gpu.lower() not in {"", "none"}:
        directives.append(f"#SBATCH --mem-per-gpu={args.slurm_mem_per_gpu}")
    if args.slurm_partition:
        directives.append(f"#SBATCH --partition={args.slurm_partition}")
    if args.slurm_account:
        directives.append(f"#SBATCH --account={args.slurm_account}")
    if args.slurm_qos:
        directives.append(f"#SBATCH --qos={args.slurm_qos}")
    if args.slurm_constraint:
        directives.append(f"#SBATCH --constraint={args.slurm_constraint}")
    for extra in args.slurm_extra_sbatch:
        extra = extra.removeprefix("#SBATCH").strip()
        directives.append(f"#SBATCH {extra}")
    return directives


def write_slurm_script(path, args, job_name, cmd, meta, mem, status_path, process_log_path):
    cmd_text = shell_join(cmd)
    python_cmd = shlex.quote(str(cmd[0]))
    run_cmd = f"srun {cmd_text}" if args.slurm_use_srun else cmd_text
    env_commands = "\n".join(args.slurm_env_command)
    if env_commands:
        env_commands += "\n"

    lines = [
        *slurm_directives(args, job_name, mem),
        f"#SBATCH --output={process_log_path}",
        f"#SBATCH --error={process_log_path}",
        "",
        "set -u",
        f"cd {shlex.quote(str(RUN_DIR))}",
        "export PYTHONUNBUFFERED=1",
        env_commands.rstrip(),
        f"export STATUS_PATH={shlex.quote(str(status_path))}",
        f"export RUN_CMD={shlex.quote(cmd_text)}",
        "export START_TIME=$(date +%s)",
        "echo \"[start] $(date)\"",
        "echo \"$RUN_CMD\"",
        run_cmd,
        "export RC=$?",
        "export END_TIME=$(date +%s)",
        "echo \"[end] $(date)\"",
        "echo \"[returncode] $RC\"",
        f"{python_cmd} - <<'PY'",
        "import json, os",
        "path = os.environ['STATUS_PATH']",
        "with open(path, 'r') as f:",
        "    data = json.load(f)",
        "start_time = int(os.environ['START_TIME'])",
        "end_time = int(os.environ['END_TIME'])",
        "data.update({",
        "    'status': 'completed',",
        "    'returncode': int(os.environ['RC']),",
        "    'start_time': start_time,",
        "    'end_time': end_time,",
        "    'elapsed_seconds': end_time - start_time,",
        "})",
        "with open(path, 'w') as f:",
        "    json.dump(data, f, indent=4)",
        "PY",
        "exit $RC",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(line for line in lines if line is not None))
    meta["slurm_script"] = str(path)


def active_slurm_jobs(job_ids):
    if not job_ids:
        return 0
    total = 0
    for i in range(0, len(job_ids), 200):
        chunk = job_ids[i:i + 200]
        result = subprocess.run(
            ["squeue", "-h", "-j", ",".join(chunk)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            total += len([line for line in result.stdout.splitlines() if line.strip()])
    return total


def wait_for_slurm_slot(job_ids, max_jobs, poll_seconds, sweep_log):
    if max_jobs <= 0:
        return
    while active_slurm_jobs(job_ids) >= max_jobs:
        append_sweep_log(sweep_log, f"[wait] {max_jobs} submitted jobs still active")
        time.sleep(poll_seconds)


def submit_slurm_job(script_path):
    result = subprocess.run(
        ["sbatch", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr}")
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    job_id = match.group(1) if match else result.stdout.strip()
    return job_id, result.stdout.strip()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=str, default="local", choices=("local", "slurm"))
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--sweep-name", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--runs-per-ablation", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--log-root", type=Path, default=Path("logs/ablations"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/ablations"))
    parser.add_argument("--start-at", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--slurm-time", type=str, default="24:00:00")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=8)
    parser.add_argument("--slurm-mem", type=str, default=None)
    parser.add_argument("--slurm-mem-buffer", type=float, default=1.5)
    parser.add_argument("--slurm-min-mem-gb", type=int, default=0)
    parser.add_argument("--slurm-vocab-size", type=int, default=12972)
    parser.add_argument("--slurm-target-vocab-size", type=int, default=2315)
    parser.add_argument("--slurm-gpu-directive", type=str, default="--gres=gpu:1")
    parser.add_argument("--slurm-mem-per-gpu", type=str, default="20G")
    parser.add_argument("--slurm-partition", type=str, default=None)
    parser.add_argument("--slurm-account", type=str, default=None)
    parser.add_argument("--slurm-qos", type=str, default=None)
    parser.add_argument("--slurm-constraint", type=str, default=None)
    parser.add_argument("--slurm-extra-sbatch", action="append", default=[])
    parser.add_argument("--slurm-env-command", action="append", default=[])
    parser.add_argument("--slurm-use-srun", type=parse_bool, default=True)
    parser.add_argument("--slurm-poll-seconds", type=int, default=60)
    return parser


def main():
    args = build_parser().parse_args()
    sweep_log = RUN_DIR / args.log_root / args.sweep_name / "sweep.log"
    env = os.environ.copy()
    if args.launcher == "local":
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONUNBUFFERED"] = "1"
    submitted_job_ids = []

    started = args.start_at is None
    for ablation_idx, ablation in enumerate(ABLATIONS):
        name = ablation["name"]
        if not started:
            started = name == args.start_at
            if not started:
                continue

        ablation_args = ablation["args"]
        ablation_slug = slugify(name)
        for run_idx in range(args.runs_per_ablation):
            seed = args.base_seed + 1000 * ablation_idx + run_idx
            run_name = f"run_{run_idx:02d}_seed_{seed}"
            run_dir = Path(ablation_slug) / run_name
            log_dir = args.log_root / args.sweep_name / run_dir
            checkpoint_dir = args.checkpoint_root / args.sweep_name / run_dir
            process_log_path = RUN_DIR / log_dir / "process.log"
            status_path = RUN_DIR / log_dir / "status.json"

            if args.skip_completed and status_path.exists():
                with status_path.open("r") as f:
                    status = json.load(f)
                if status.get("returncode") == 0:
                    append_sweep_log(sweep_log, f"[skip] {name} {run_name}")
                    continue

            cmd = [
                args.python,
                "run.py",
                "--device", args.device,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--save-every", str(args.epochs),
                "--log-dir", str(log_dir),
                "--checkpoint-dir", str(checkpoint_dir),
            ]
            add_cli_args(cmd, ablation_args)

            if args.dry_run:
                print(f"[dry-run] {name} {run_name}", flush=True)
                print(shell_join(cmd), flush=True)
                continue

            job_slug = slugify(f"wb_{ablation_slug}_{run_idx}")
            job_name = job_slug[:128]
            mem = slurm_mem(args, ablation_args)
            slurm_script_path = RUN_DIR / log_dir / "job.sbatch"
            meta = {
                "ablation": name,
                "ablation_args": ablation_args,
                "run_idx": run_idx,
                "seed": seed,
                "epochs": args.epochs,
                "launcher": args.launcher,
                "gpu": args.gpu if args.launcher == "local" else None,
                "cmd": cmd,
                "log_dir": str(log_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "slurm_mem": mem if args.launcher == "slurm" else None,
            }
            write_json(RUN_DIR / log_dir / "metadata.json", meta)

            if args.launcher == "slurm":
                process_log_path.parent.mkdir(parents=True, exist_ok=True)
                status = {
                    **meta,
                    "status": "prepared",
                    "returncode": None,
                    "slurm_job_id": None,
                    "slurm_script": str(slurm_script_path),
                }
                write_json(status_path, status)
                write_slurm_script(
                    slurm_script_path,
                    args,
                    job_name,
                    cmd,
                    meta,
                    mem,
                    RUN_DIR / status_path,
                    process_log_path,
                )
                write_json(RUN_DIR / log_dir / "metadata.json", meta)
                wait_for_slurm_slot(
                    submitted_job_ids,
                    args.max_jobs,
                    args.slurm_poll_seconds,
                    sweep_log,
                )
                job_id, submit_message = submit_slurm_job(slurm_script_path)
                submitted_job_ids.append(job_id)
                status.update({
                    "status": "submitted",
                    "slurm_job_id": job_id,
                    "submit_message": submit_message,
                    "submit_time": time.time(),
                })
                write_json(status_path, status)
                append_sweep_log(sweep_log, f"[submit] {name} {run_name} job_id={job_id}")
                append_sweep_log(sweep_log, shell_join(["sbatch", slurm_script_path]))
                continue

            append_sweep_log(sweep_log, f"[start] {name} {run_name}")
            append_sweep_log(sweep_log, shell_join(cmd))

            start_time = time.time()
            process_log_path.parent.mkdir(parents=True, exist_ok=True)
            with process_log_path.open("a") as process_log:
                process_log.write(f"[start] {time.ctime(start_time)}\n")
                process_log.write(shell_join(cmd) + "\n")
                process_log.flush()
                result = subprocess.run(
                    cmd,
                    cwd=RUN_DIR,
                    env=env,
                    stdout=process_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                end_time = time.time()
                process_log.write(f"[end] {time.ctime(end_time)}\n")
                process_log.write(f"[returncode] {result.returncode}\n")

            status = {
                **meta,
                "start_time": start_time,
                "end_time": end_time,
                "elapsed_seconds": end_time - start_time,
                "returncode": result.returncode,
            }
            write_json(status_path, status)

            if result.returncode != 0:
                append_sweep_log(sweep_log, f"[fail] {name} {run_name} rc={result.returncode}")
                if not args.continue_on_error:
                    raise SystemExit(result.returncode)
            append_sweep_log(sweep_log, f"[done] {name} {run_name}")

    if args.start_at is not None and not started:
        raise SystemExit(f"--start-at {args.start_at!r} did not match any ablation")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.stdout = open(os.devnull, "w")
