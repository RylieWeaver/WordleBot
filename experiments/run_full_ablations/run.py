# General
import argparse
import random
from pathlib import Path
import numpy as np

# Torch
import torch

# Wordle
from wordle.data import get_vocab, words_to_tensor, WordleLoaderConfig
from wordle.environment import SimulatorConfig
from wordle.model import (
    ActorCriticNetConfig, ActorCriticNet,
    DotGuessStateNetConfig, DotGuessStateNet,
    WordleTransformerConfig, WordleTransformer,
)
from wordle.train import (
    WordleLossConfig, OptHandlerConfig, SchedulerConfig,
    LoggerConfig, TrainerConfig, Trainer
)
from wordle.utils import resolve_device


RUN_DIR = Path(__file__).resolve().parent


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def parse_reduce_dims(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"none", "null"}:
        return None
    if value == "":
        return ()
    return tuple(int(dim.strip()) for dim in value.split(",") if dim.strip())


def resolve_run_path(path):
    path = Path(path)
    return path if path.is_absolute() else RUN_DIR / path


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)

    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        default="DotGuessStateNet",
        choices=("ActorCriticNet", "DotGuessStateNet", "WordleTransformer"),
    )
    parser.add_argument("--model-size-multiplier", type=float, default=4.0)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-inductive-biases", type=parse_bool, default=True)

    # Loader / simulator
    parser.add_argument("--loader-batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-guesses", type=int, default=6)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--advantage-type", type=str, default="gae")
    parser.add_argument("--adv-mean-reduce-dims", type=parse_reduce_dims, default=(2,))
    parser.add_argument("--adv-std-reduce-dims", type=parse_reduce_dims, default=(0, 2))
    parser.add_argument("--correct-reward", type=float, default=0.1)
    parser.add_argument("--correct-blend-factor", type=float, default=1.0)
    parser.add_argument("--reward-blend-factor", type=float, default=1.0)
    parser.add_argument("--value-blend-factor", type=float, default=1.0)

    # Loss
    parser.add_argument("--actor-loss-weight", type=float, default=5.0)
    parser.add_argument("--critic-loss-weight", type=float, default=5.0)
    parser.add_argument("--entropy-loss-weight", type=float, default=0.05)
    parser.add_argument("--kl-reg-loss-weight", type=float, default=1.0)
    parser.add_argument("--kl-guide-loss-weight", type=float, default=0.25)
    parser.add_argument("--kl-best-loss-weight", type=float, default=0.10)
    parser.add_argument("--ratio-prob-clip", type=float, default=0.2)

    # Optimizer / scheduler
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--init-alpha", type=float, default=0.25)
    parser.add_argument("--min-alpha", type=float, default=0.0)
    parser.add_argument("--alpha-step", type=float, default=0.16)
    parser.add_argument("--init-temperature", type=float, default=1.0)
    parser.add_argument("--min-temperature", type=float, default=0.30)
    parser.add_argument("--temperature-decay", type=float, default=0.90)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=1000)

    # Trainer
    parser.add_argument("--processing-batch-size", type=int, default=16)
    parser.add_argument("--batches-per-gradient-step", type=int, default=16)
    parser.add_argument("--rollout-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=("none", "bfloat16"))
    parser.add_argument("--rest-computer", type=float, default=0.0)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/test"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/test"))
    return parser


def build_models(args, total_vocab, total_vocab_tensor, device):
    c = args.model_size_multiplier
    common_kwargs = {
        "layers": args.layers,
        "dropout": args.dropout,
        "use_inductive_biases": args.use_inductive_biases,
    }

    if args.model_name == "ActorCriticNet":
        model_cfg = ActorCriticNetConfig(
            hidden_dim=int(128 * c),
            output_dim=len(total_vocab),
            **common_kwargs,
        )
        model = ActorCriticNet(model_cfg, device=device).to(device)
        ref_model = ActorCriticNet(model_cfg, device=device).to(device).eval()
        best_model = ActorCriticNet(model_cfg, device=device).to(device).eval()
    elif args.model_name == "DotGuessStateNet":
        model_cfg = DotGuessStateNetConfig(
            state_hidden_dim=int(128 * c),
            guess_hidden_dim=int(8 * c),
            output_dim=int(128 * c),
            vocab_size=len(total_vocab),
            **common_kwargs,
        )
        model = DotGuessStateNet(model_cfg, total_vocab_tensor=total_vocab_tensor, device=device).to(device)
        ref_model = DotGuessStateNet(model_cfg, total_vocab_tensor=total_vocab_tensor, device=device).to(device).eval()
        best_model = DotGuessStateNet(model_cfg, total_vocab_tensor=total_vocab_tensor, device=device).to(device).eval()
    elif args.model_name == "WordleTransformer":
        model_cfg = WordleTransformerConfig(
            state_hidden_dim=int(128 * c),
            guess_hidden_dim=int(8 * c),
            output_dim=int(128 * c),
            vocab_size=len(total_vocab),
            **common_kwargs,
        )
        model = WordleTransformer(model_cfg, total_vocab_tensor=total_vocab_tensor, device=device).to(device)
        ref_model = WordleTransformer(model_cfg, total_vocab_tensor=total_vocab_tensor, device=device).to(device).eval()
        best_model = WordleTransformer(model_cfg, total_vocab_tensor=total_vocab_tensor, device=device).to(device).eval()
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    ref_model.load_state_dict(model.state_dict())
    best_model.load_state_dict(model.state_dict())
    return ref_model, model, best_model


def log_model_stats(trainer):
    model = trainer.model
    num_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer.logger.log(f"Model: {model.cfg.model_name}")
    trainer.logger.log(f"Model Parameters: {num_parameters:,}")
    trainer.logger.log(f"Trainable Parameters: {trainable_parameters:,}")
    trainer.logger.log(f"Forward FLOPs: {model.forward_flops():,}")
    trainer.logger.log(f"Backward FLOPs: {model.backward_flops():,}")
    trainer.logger.log(f"AMP dtype: {trainer.cfg.amp_dtype}")


def build_fresh_trainer(args, device):
    target_vocab = get_vocab(vocab_type="target")
    nontarget_vocab = get_vocab(vocab_type="nontarget")
    total_vocab = np.concatenate((target_vocab, nontarget_vocab), axis=0)

    total_vocab_tensor = words_to_tensor(total_vocab).to(device)
    ref_model, model, best_model = build_models(args, total_vocab, total_vocab_tensor, device)

    loader_cfg = WordleLoaderConfig(
        target_vocab=target_vocab,
        nontarget_vocab=nontarget_vocab,
        batch_size=args.loader_batch_size,
        repeats=args.repeats,
        shuffle=True,
        num_workers=args.num_workers,
    )
    simulator_cfg = SimulatorConfig(
        loader_cfg=loader_cfg,
        gamma=args.gamma,
        lam=args.lam,
        max_guesses=args.max_guesses,
        m=args.m,
        correct_reward=args.correct_reward,
        correct_blend_factor=args.correct_blend_factor,
        reward_blend_factor=args.reward_blend_factor,
        value_blend_factor=args.value_blend_factor,
        advantage_type=args.advantage_type,
        adv_mean_reduce_dims=args.adv_mean_reduce_dims,
        adv_std_reduce_dims=args.adv_std_reduce_dims,
    )
    loss_cfg = WordleLossConfig(
        loss_weights={
            "actor": args.actor_loss_weight,
            "critic": args.critic_loss_weight,
            "entropy": args.entropy_loss_weight,
            "kl_reg": args.kl_reg_loss_weight,
            "kl_guide": args.kl_guide_loss_weight,
            "kl_best": args.kl_best_loss_weight,
        },
        ratio_prob_clip=args.ratio_prob_clip,
    )
    opt_handler_cfg = OptHandlerConfig(
        name="adamw",
        lr=args.learning_rate,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
    )
    scheduler_cfg = SchedulerConfig(
        init_alpha=args.init_alpha,
        min_alpha=args.min_alpha,
        alpha_step=args.alpha_step,
        init_temperature=args.init_temperature,
        min_temperature=args.min_temperature,
        temperature_decay=args.temperature_decay,
        patience=args.patience,
        warmup_steps=args.warmup_steps,
    )
    logger_cfg = LoggerConfig(
        log_dir=resolve_run_path(args.log_dir),
    )
    trainer_cfg = TrainerConfig(
        processing_batch_size=args.processing_batch_size,
        batches_per_gradient_step=args.batches_per_gradient_step,
        rollout_size=args.rollout_size,
        simulator_cfg=simulator_cfg,
        loss_cfg=loss_cfg,
        opt_handler_cfg=opt_handler_cfg,
        scheduler_cfg=scheduler_cfg,
        logger_cfg=logger_cfg,
        checkpoint_dir=resolve_run_path(args.checkpoint_dir),
        save_every=args.save_every,
        amp_dtype=args.amp_dtype,
        rest_computer=args.rest_computer,
    )
    return Trainer(trainer_cfg, ref_model, model, best_model, device=device)


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint_dir = resolve_run_path(args.checkpoint_dir)

    if args.resume_from:
        trainer = Trainer.load_checkpoint(checkpoint_dir / args.resume_from, device=device)
    else:
        trainer = build_fresh_trainer(args, device)

    log_model_stats(trainer)
    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
