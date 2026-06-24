# General
import argparse
from pathlib import Path
import numpy as np

# Torch

# Wordle
from wordle.data import get_vocab, words_to_tensor, WordleLoaderConfig
from wordle.environment import SimulatorConfig
from wordle.model import DotGuessStateNetConfig, DotGuessStateNet, ActorCriticNetConfig, ActorCriticNet
from wordle.train import (
    WordleLossConfig, OptHandlerConfig, SchedulerConfig, 
    LoggerConfig, TrainerConfig, Trainer
)
from wordle.utils import resolve_device


# Notes:
# - Change comments if loading from a checkpoint or fresh start



if __name__ == '__main__':
    # Read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=None, help="Path to directory")
    args = parser.parse_args()

    # Setup
    device = resolve_device("cuda")

    # Train from scratch
    if not args.resume_from:
        # Data
        # target_vocab = get_vocab(vocab_type="target", size=200)                     # [T]
        # nontarget_vocab = get_vocab(vocab_type="nontarget", size=800)               # [V-T]
        target_vocab = get_vocab(vocab_type="target")                               # [T]
        nontarget_vocab = get_vocab(vocab_type="nontarget")                         # [V-T]
        total_vocab = np.concatenate((target_vocab, nontarget_vocab), axis=0)       # [V]

        # Model
        c = 4
        model_cfg = DotGuessStateNetConfig(
            # state_input_dim = 292 = 26 letters * 11 letter possibilities (1 count, 5 green, 5 grey) + 6 (one-hot of guess number)
            state_hidden_dim=128 * c,
            guess_hidden_dim=8 * c,
            output_dim=128 * c,
            vocab_size=len(total_vocab),
            layers=3,
            dropout=0.0,
            use_inductive_biases=True,
        )
        total_vocab_tensor = words_to_tensor(total_vocab).to(device)                # [V, 5]
        model = DotGuessStateNet(model_cfg, total_vocab_tensor=total_vocab_tensor, device=device).to(device)
        ref_model = DotGuessStateNet(model_cfg, total_vocab_tensor=total_vocab_tensor, device=device).to(device).eval()
        best_model = DotGuessStateNet(model_cfg, total_vocab_tensor=total_vocab_tensor, device=device).to(device).eval()
        # model_cfg = ActorCriticNetConfig(
        #     # state_input_dim = 292 = 26 letters * 11 letter possibilities (1 count, 5 green, 5 grey) + 6 (one-hot of guess number)
        #     hidden_dim=128 * c,
        #     output_dim=len(total_vocab),
        #     layers=3,
        #     dropout=0.0,
        #     use_inductive_biases=True,
        # )
        # model = ActorCriticNet(model_cfg, device=device).to(device)
        # # model.load_state_dict(torch.load(f'{load_dir}/model.pth', map_location=device, weights_only=True))
        # ref_model = ActorCriticNet(model_cfg, device=device).to(device).eval()
        # best_model = ActorCriticNet(model_cfg, device=device).to(device).eval()
        ref_model.load_state_dict(model.state_dict())
        best_model.load_state_dict(model.state_dict())

        # Configs fed into trainer
        loader_cfg = WordleLoaderConfig(
            target_vocab=target_vocab,
            nontarget_vocab=nontarget_vocab,
            batch_size=32,
            repeats=32,
            shuffle=True,
            num_workers=4,
        )
        simulator_cfg = SimulatorConfig(
            loader_cfg=loader_cfg,
            max_guesses=6,
            m=32,
            num_search_actions=10,
        )
        loss_cfg = WordleLossConfig(
            loss_weights={
                "actor": 1.0,
                "critic": 0.0,
                "entropy": 0.0,
                "kl_reg": 0.0,
                "kl_guide": 0.0,
            },
            ratio_prob_clip=0.2,
        )
        opt_handler_cfg = OptHandlerConfig(
            name="adamw",
            lr=3e-4,
            weight_decay=1e-4,
        )
        scheduler_cfg = SchedulerConfig(
            init_alpha=0.09,
            min_alpha=0.00,
            alpha_step=0.16,
            init_temperature=1.00,
            min_temperature=0.30,
            temperature_decay=0.90,
            patience=3,
            warmup_steps=300,
        )
        logger_cfg = LoggerConfig(
            log_dir=Path("logs/test"),
        )
        # Trainer
        trainer_cfg = TrainerConfig(
            processing_batch_size=5,
            batches_per_gradient_step=463,
            rollout_size=6,
            simulator_cfg=simulator_cfg,
            loss_cfg=loss_cfg,
            opt_handler_cfg=opt_handler_cfg,
            scheduler_cfg=scheduler_cfg,
            logger_cfg=logger_cfg,
            checkpoint_dir=Path("checkpoints/test"),
            save_every=200,
            amp_dtype="bfloat16",
            rest_computer=0.0,
        )
        trainer = Trainer(trainer_cfg, ref_model, model, best_model, device=device)
    # Trainer from checkpoint
    else:
        ckpt_dir = Path(f"checkpoints/test/{args.resume_from}")
        trainer = Trainer.load_checkpoint(ckpt_dir, device=device)
    
    # Train the network
    trainer.train(epochs=200)
