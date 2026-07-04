# Run Full Ablations

Run all commands from this directory:

```bash
cd */WordleBot/experiments/run_full_ablations
```

If needed, activate the environment first:

```bash
source <env-path>
```


## Prepare Data

Generate the local vocab files used by `run.py`:

```bash
python get_data.py
```


## Quick Test

Run the full sweep plan without launching training:

```bash
python run_sweep.py --dry-run --epochs 5
```

Run one local sequential smoke test on a single visible GPU:

```bash
python run_sweep.py \
  --launcher local \
  --gpu 0 \
  --epochs 5 \
  --runs-per-ablation 3 \
  --continue-on-error
```


## Memory Estimate

Estimate memory first:

```bash
python memory_estimate.py --sweep --csv > memory_estimates.csv
```


## Local Sweep

This runs one job at a time on one GPU:

```bash
python run_sweep.py \
  --launcher local \
  --gpu 0 \
  --epochs 200 \
  --runs-per-ablation 3 \
  --continue-on-error \
  --skip-completed
```

To run several local GPUs from one scheduler process, pass a comma-separated
GPU list. Each child process sees its assigned GPU as `cuda:0`.

```bash
nohup python3 run_sweep.py \
  --launcher local \
  --gpus 4,5,6,7 \
  --epochs 200 \
  --runs-per-ablation 3 \
  --processing-num-workers 4 \
  --continue-on-error \
  --skip-completed \
  > nohup.out 2>&1 &
```

`--num-workers` controls the rollout data loader. `--processing-num-workers`
controls the `EpisodesDataset` loaders used for eval and rollout processing.
The processing worker default is `4`.

Outputs go under:

- `logs/ablations/<sweep-name>/<ablation>/run_XX_seed_YYYY/`
- `checkpoints/ablations/<sweep-name>/<ablation>/run_XX_seed_YYYY/`

By default, the sweep passes `--save-every <epochs>`, so the scheduled checkpoint
is at the final epoch for each run. Ablation runs also default to
`--save-best false`, which keeps in-memory best model updates while avoiding
best-checkpoint writes. Pass `--save-best true` to write full checkpoints when a
new best eval is found.

- best checkpoints use the normal `epoch_N/` directory format
- final checkpoints use the same `epoch_N/` format, usually `epoch_200/`

Use `--sweep-name my_name` to make the output directory explicit.


## Slurm Sweep

Submit one Slurm job per ablation seed/run:

```bash
python run_sweep.py \
  --launcher slurm \
  --epochs 200 \
  --runs-per-ablation 3 \
  --max-jobs 0 \
  --slurm-account YOUR_PROJECT \
  --slurm-partition YOUR_PARTITION \
  --slurm-qos YOUR_QOS \
  --slurm-time 24:00:00 \
  --slurm-cpus-per-task 8 \
  --slurm-mem 90G \
  --slurm-pythonpath /path/to/WordleBot \
  --slurm-env-command "source <env-path>"
```

These map to Slurm's `-A`, `-p`, and `-q` fields:

- `--slurm-account YOUR_PROJECT` writes `#SBATCH --account=YOUR_PROJECT`
- `--slurm-partition YOUR_PARTITION` writes `#SBATCH --partition=YOUR_PARTITION`
- `--slurm-qos YOUR_QOS` writes `#SBATCH --qos=YOUR_QOS`

By default, generated jobs request regular Slurm memory with `#SBATCH --mem=...`
using the estimator in `memory_estimate.py`. Override it with `--slurm-mem 90G`
if needed.

Use `--slurm-pythonpath /path/to/WordleBot` if the repo is not installed in the
activated environment. Multiple `--slurm-pythonpath` values are joined with `:`.

Cluster-specific examples to adjust as needed:

```bash
python run_sweep.py \
  --launcher slurm \
  --slurm-account YOUR_ACCOUNT \
  --slurm-partition gpu \
  --slurm-qos normal \
  --slurm-gpu-directive "--gres=gpu:1" \
  --slurm-pythonpath /path/to/WordleBot \
  --slurm-extra-sbatch "--mail-type=END,FAIL"
```

Set `--max-jobs N` to throttle how many submitted jobs may be active at once.


## Resume / Partial Runs

Skip runs that already completed successfully:

```bash
python run_sweep.py --launcher local --skip-completed
```

Start at a specific ablation name:

```bash
python run_sweep.py --start-at model_size_4
```

Resume a run from a checkpoint:

```bash
python run.py --resume-from epoch_200 --checkpoint-dir checkpoints/ablations/<sweep-name>/<ablation>/run_XX_seed_YYYY
```

Each run writes:

- `metadata.json`: command, seed, ablation args
- `status.json`: completion status and return code
- `process.log`: stdout/stderr for that run
- `sweep.log`: top-level launcher progress
