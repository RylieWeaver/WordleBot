# General
import gc
import time, json
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union
import traceback
from contextlib import nullcontext

# Torch
import torch
from torch.utils.data import DataLoader

# Wordle
from wordle.model import name2model
from wordle.data import EpisodesDataset, tensor_to_words, words_to_tensor, move_to
from wordle.environment import SimulatorConfig, Simulator
from wordle.train import OptHandlerConfig, OptHandler, SchedulerConfig, Scheduler, LoggerConfig, Logger, WordleLossConfig, WordleLoss
from wordle.utils import clear_cache, Config



class TrainerConfig(Config):
    def __init__(
            self,
            processing_batch_size: int = 32,
            batches_per_gradient_step: int = 1,
            rollout_size: int = 4,
            simulator_cfg: SimulatorConfig = None,
            loss_cfg: WordleLossConfig = None,
            opt_handler_cfg: OptHandlerConfig = None,
            scheduler_cfg: SchedulerConfig = None,
            logger_cfg: LoggerConfig = None,
            checkpoint_dir: Optional[Union[Path, str]] = None,
            save_every: Optional[int] = None,
            fp_dtype: str = "float32",
            amp_dtype: str = "none",
            rest_computer: Optional[float] = 1e-3,
            max_batch_attempts: int = 3,
    ) -> None:
        # Vars
        self.batches_per_gradient_step = batches_per_gradient_step
        self.processing_batch_size = processing_batch_size
        self.rollout_size = rollout_size
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.fp_dtype = fp_dtype
        self.amp_dtype = amp_dtype
        self.rest_computer = rest_computer
        self.max_batch_attempts = max_batch_attempts

        # Objects
        self.simulator_cfg = simulator_cfg if simulator_cfg is not None else SimulatorConfig()
        self.loss_cfg = loss_cfg if loss_cfg is not None else WordleLossConfig()
        self.opt_handler_cfg = opt_handler_cfg if opt_handler_cfg is not None else OptHandlerConfig()
        self.scheduler_cfg = scheduler_cfg if scheduler_cfg is not None else SchedulerConfig()
        self.logger_cfg = logger_cfg if logger_cfg is not None else LoggerConfig()

    @staticmethod
    def load(path: Union[str, Path]) -> "TrainerConfig":
        """
        The keys here for OptHandler, LoaderHandler, Logger , etc...
        must match the varnames of their respective instances.
        """
        path = Path(path)
        with path.open("r") as f:
            cfg = json.load(f)
        return TrainerConfig(
            processing_batch_size=cfg.get("processing_batch_size", 1),
            batches_per_gradient_step=cfg.get("batches_per_gradient_step", 1),
            rollout_size=cfg.get("rollout_size", 4),
            simulator_cfg=SimulatorConfig(**cfg.get("simulator_cfg", {})),
            loss_cfg=WordleLossConfig(**cfg.get("loss_cfg", {})),
            opt_handler_cfg=OptHandlerConfig(**cfg.get("opt_handler_cfg", {})),
            scheduler_cfg=SchedulerConfig(**cfg.get("scheduler_cfg", {})),
            logger_cfg=LoggerConfig(**cfg.get("logger_cfg", {})),
            checkpoint_dir=cfg.get("checkpoint_dir", None),
            save_every=cfg.get("save_every", None),
            fp_dtype=cfg.get("fp_dtype", "float32"),
            amp_dtype=cfg.get("amp_dtype", "none"),
            rest_computer=cfg.get("rest_computer", 1e-3),
            max_batch_attempts=cfg.get("max_batch_attempts", 3),
        )


class Trainer:
    def __init__(self, cfg, ref_model, model, best_model, device: Union[str, torch.device] = "cpu") -> None:
        # Read vars
        self.cfg = cfg
        self.ref_model = ref_model.to(device).eval()
        self.model = model.to(device)
        self.best_model = best_model.to(device).eval()
        self.device = device
        self.fp_dtype = getattr(torch, self.cfg.fp_dtype)
        self.amp_dtype = None if self.cfg.amp_dtype in {None, "none"} else getattr(torch, self.cfg.amp_dtype)
        self.amp_device_type = torch.device(device).type
        self.use_amp = (self.amp_dtype is not None) and (self.amp_device_type == "cuda")

        # Build objects
        self.simulator = Simulator(self.cfg.simulator_cfg)
        self.loss = WordleLoss(self.cfg.loss_cfg)
        self.opt_handler = OptHandler(self.cfg.opt_handler_cfg)
        self.optimizer = self.opt_handler.build_optimizer(self.model.parameters())
        self.scheduler = Scheduler(self.optimizer, self.cfg.scheduler_cfg)
        self.logger = Logger(self.cfg.logger_cfg)

        # Init trainer state
        self.last_epoch = 0
        self.best_accuracy = -1e-8
        self.best_avg_guesses = float('inf')

    """
    The `__reinit__` method can be useful to make align objects that are shared 
    between trainer and its config (e.g. fp_dtype). Usually this happens when 
    overwriting something after loading from checkpoint.
    """
    def __reinit__(self):
        self.__init__(self.cfg, self.ref_model, self.model, self.best_model, self.device)

    def state_dict(self):
        return {
            "last_epoch": self.last_epoch,
            "best_accuracy": self.best_accuracy,
            "best_avg_guesses": self.best_avg_guesses,
        }
    
    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict["last_epoch"]
        self.best_accuracy = state_dict["best_accuracy"]
        self.best_avg_guesses = state_dict["best_avg_guesses"]

    def _amp_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type=self.amp_device_type, dtype=self.amp_dtype)

    def _float32_tree(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.float() if obj.is_floating_point() else obj
        if isinstance(obj, dict):
            return {k: self._float32_tree(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._float32_tree(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._float32_tree(v) for v in obj)
        return obj

    def save_checkpoint(self, epoch: int):
        # Setup
        save_dir = Path(self.cfg.checkpoint_dir) / f"epoch_{epoch}" if self.cfg.checkpoint_dir else None
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save all other configs (simulator, loss, opt_handler, scheduler, logger)
        self.simulator.loader.cfg.save(save_dir / "loader_config.json")
        self.simulator.cfg.save(save_dir / "simulator_config.json")
        self.loss.cfg.save(save_dir / "loss_config.json")
        self.opt_handler.cfg.save(save_dir / "opt_handler_config.json")
        self.scheduler.cfg.save(save_dir / "scheduler_config.json")
        self.logger.cfg.save(save_dir / "logger_config.json")

        # Save models (ref, current, best)
        self.ref_model.cfg.save(save_dir / "ref_model_config.json")
        torch.save(self.ref_model.state_dict(), save_dir / "ref_model.pt")
        self.model.cfg.save(save_dir / "model_config.json")
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        self.best_model.cfg.save(save_dir / "best_model_config.json")
        torch.save(self.best_model.state_dict(), save_dir / "best_model.pt")

        # Save trainer
        self.cfg.save(save_dir / "trainer_config.json")
        torch.save(self.state_dict(), save_dir / "trainer.pt")

        # Save optimizer
        # NOTE: Optimizer config is nested and saved within trainer config because .to_dict() is recursive
        torch.save(self.optimizer.state_dict(), save_dir / "optimizer.pt")

        # Save scheduler
        # NOTE: Scheduler config is nested and saved within trainer config because .to_dict() is recursive
        torch.save(self.scheduler.state_dict(), save_dir / "scheduler.pt")
    
    @staticmethod
    def load_checkpoint(load_dir: Union[Path, str], device: torch.device) -> "Trainer":
        # Setup
        load_dir = Path(load_dir)

        # Helper function since we load multiple models
        def load_model_ckpt(model_cfg_path: Path, model_state_path: Path, device=None):
            # Load model class dynamically from name2model
            with model_cfg_path.open("r") as f:
                model_cfg = json.load(f)
            # NOTE: We pop out model name so that it's not passed, even though **kwargs should catch it
            model_cfg_cls, model_cls = name2model[model_cfg.pop("model_name")]
            model_cfg = model_cfg_cls(**model_cfg)
            model = model_cls(model_cfg, device=device).to(device)
            state_dict = torch.load(model_state_path, weights_only=True)
            model.load_state_dict(state_dict)
            return model

        # Load models (ref, current, best)
        ref_model = load_model_ckpt(
            model_cfg_path=(load_dir / "ref_model_config.json"),
            model_state_path=(load_dir / "ref_model.pt"),
            device=device
        )
        for param in ref_model.parameters():
            param.requires_grad = False
        model = load_model_ckpt(
            model_cfg_path=(load_dir / "model_config.json"),
            model_state_path=(load_dir / "model.pt"),
            device=device
        )
        best_model = load_model_ckpt(
            model_cfg_path=(load_dir / "best_model_config.json"),
            model_state_path=(load_dir / "best_model.pt"),
            device=device
        )
        for param in best_model.parameters():
            param.requires_grad = False

        # Load trainer
        trainer_cfg = TrainerConfig.load(load_dir / "trainer_config.json")
        trainer = Trainer(trainer_cfg, ref_model, model, best_model, device)
        trainer_state = torch.load(load_dir / "trainer.pt", map_location=device)
        trainer.load_state_dict(trainer_state)

        # Load optimizer
        trainer.optimizer = trainer.opt_handler.build_optimizer(trainer.model.parameters())
        optimizer_state = torch.load(load_dir / "optimizer.pt", map_location=device)
        trainer.optimizer.load_state_dict(optimizer_state)

        # Load scheduler (make sure to link the loaded optimizer to the scheduler)
        trainer.scheduler = Scheduler(trainer.optimizer, trainer.cfg.scheduler_cfg)
        scheduler_state = torch.load(load_dir / "scheduler.pt", map_location=device)
        trainer.scheduler.load_state_dict(scheduler_state)
        return trainer

    def _run_batch(self, episodes, alpha=None, temp=None, measure_grad_norms=False):
        for attempt in range(1, self.cfg.max_batch_attempts + 1):
            try:
                # Setup (quicker reference vars)
                alpha = self.scheduler.alpha if alpha is None else alpha
                temp = self.scheduler.temperature if temp is None else temp
                states = episodes["states"]
                actions = episodes["actions"]

                # Forward episodes through models
                episodes = move_to(episodes, self.model.device)
                with self._amp_context():
                    probs, responses = self.simulator.process_episodes(self.model, episodes, alpha, temp)
                    with torch.no_grad():
                        states = move_to(states, self.ref_model.device)
                        ref_probs, _ = self.ref_model.predict(states, alpha, temp)
                        states = move_to(states, self.best_model.device)
                        best_probs, _ = self.best_model.predict(states, alpha, temp)
                
                # Increment loss
                states = move_to(states, self.device)
                actions = move_to(actions, self.device)
                responses = self._float32_tree(move_to(responses, self.device))
                probs = self._float32_tree(move_to(probs, self.device))
                ref_probs = self._float32_tree(move_to(ref_probs, self.device))
                best_probs = self._float32_tree(move_to(best_probs, self.device))
                batch_loss, batch_loss_components = self.loss.inc_loss(
                    states, actions, responses, probs, ref_probs, best_probs
                )

                # Measure grad norms
                if measure_grad_norms:
                    self.loss.measure_grad_norms(self.model, states, actions, responses, probs, ref_probs, best_probs, alpha, temp)
                return batch_loss, batch_loss_components
            except RuntimeError as e:
                if attempt < self.cfg.max_batch_attempts:
                    print(f"Retrying episode processing due to error:\n")
                    traceback.print_exc()
                else:
                    self.optimizer.zero_grad(set_to_none=True)  # Reset optimizer state
                    clear_cache()
                    print(f"Failed to process episodes after {self.cfg.max_batch_attempts} attempts due to error:\n.")
                    traceback.print_exc()
                    continue    # out of retries, continue to raise the error
                time.sleep(3)   # wait 3 seconds before next try

    def measure_grad_norms(self, episodes_batch, alpha=None, temp=None):
        # Setup (quicker reference vars)
        alpha = self.scheduler.alpha if alpha is None else alpha
        temp = self.scheduler.temperature if temp is None else temp
        states = episodes_batch["states"]
        actions = episodes_batch["actions"]

        # Forward episodes through models
        episodes_batch = move_to(episodes_batch, self.model.device)
        with self._amp_context():
            probs, responses = self.simulator.process_episodes(self.model, episodes_batch, alpha, temp)
            with torch.no_grad():
                states = move_to(states, self.ref_model.device)
                ref_probs, _ = self.ref_model.predict(states, alpha, temp)
                states = move_to(states, self.best_model.device)
                best_probs, _ = self.best_model.predict(states, alpha, temp)
        
        # Measure grad norms
        states = move_to(states, self.device)
        actions = move_to(actions, self.device)
        responses = self._float32_tree(move_to(responses, self.device))
        probs = self._float32_tree(move_to(probs, self.device))
        ref_probs = self._float32_tree(move_to(ref_probs, self.device))
        best_probs = self._float32_tree(move_to(best_probs, self.device))
        self.loss.measure_grad_norms(self.model, states, actions, responses, probs, ref_probs, best_probs)

    def _loop_without_grad(self, loader, desc, alpha=None, temp=None):
        self.loss.init_cumulative_loss()
        with torch.no_grad():
            for episodes_batch in tqdm(loader, desc=desc, leave=False):
                self._run_batch(episodes_batch, alpha=alpha, temp=temp)
            # Rest computer to prevent overheating (rough estimate of compute load is batch_size * repeats)
            if self.cfg.rest_computer:
                time.sleep(float(self.cfg.rest_computer) * loader.batch_size * self.simulator.loader.repeats)
        num_batches = len(loader)
        self.loss.average_cumulative_loss(num_batches)
        return self.loss.loss, self.loss.loss_components
    
    def _loop_with_grad(self, loader, desc):
        self.model.train()
        self.loss.init_cumulative_loss()
        self.optimizer.zero_grad(set_to_none=True)
        num_batches = len(loader)

        for i, episodes_batch in enumerate(tqdm(loader, desc=desc, leave=False)):
            batch_loss, batch_loss_components = self._run_batch(episodes_batch)
            mb_size = episodes_batch["states"]["active_mask"].shape[0]
            batch_scale = (mb_size / loader.batch_size) * (1.0 / self.cfg.batches_per_gradient_step)
            (batch_loss * batch_scale).backward()
            # Step optimizer every N batches or at end of epoch
            if ((i + 1) % self.cfg.batches_per_gradient_step == 0) or (i + 1 == len(loader)):
                self.opt_handler.clip_grad_norm(self.model)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            # Rest computer to prevent overheating (rough estimate of compute load is batch_size * repeats)
            if self.cfg.rest_computer:
                time.sleep(float(self.cfg.rest_computer) * loader.batch_size * self.simulator.loader.repeats)
        
        # Measure grad norms on the last batch
        self.measure_grad_norms(episodes_batch)

        # Average and return
        self.loss.average_cumulative_loss(num_batches)
        return self.loss.loss, self.loss.loss_components
    
    def _run_eval(self, epoch: int):
        # Evaluate with eval mode and argmax policy
        self.model.eval()
        test_episodes = self.simulator.collect_episodes_epoch(
            self.model,
            alpha=0.0,
            temperature=1.0,
            argmax=True,
            desc="Collecting Eval Episodes"
        )
        # Evaluate stats
        active_mask = test_episodes["states"]["active_mask"][:, :-1, ...]       # [B, G, *] ignore active mask of final state after last guess
        test_wins = (
            test_episodes["responses"]["correct"].bool() & active_mask.bool()
        ).any(dim=1).float().sum().item()                                       # [B, G, *] --> [B, *]
        test_guesses = active_mask.float().sum().item()                         # [B, G, *] --> scalar
        test_games = active_mask[:, 0, ...].float().sum().item()                # [B, G, *] --> [B, *] --> scalar
        test_acc = (test_wins / test_games)
        test_avg_guesses = (test_guesses / test_games)

        # Create dataset/loader for processing eval episodes
        test_dataset = EpisodesDataset(test_episodes)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.processing_batch_size, shuffle=False, num_workers=0)
        test_loss, test_loss_components = self._loop_without_grad(loader=test_loader, desc="Evaluating", alpha=0.0, temp=1.0)
        self.logger.log(
            f"[Epoch {epoch}][Eval] Accuracy: {test_acc:.4f} | Avg Guesses: {test_avg_guesses:.4f} | "
            f"Actor Loss: {test_loss_components['actor']:.6f} | Critic Loss: {test_loss_components['critic']:.6f} | "
            f"Entropy Loss: {test_loss_components['entropy']:.6f} | KL Reg Loss: {test_loss_components['kl_reg']:.6f} | "
            f"KL Guide Loss: {test_loss_components['kl_guide']:.6f} | KL Best Loss: {test_loss_components['kl_best']:.6f} "
        )
        # Explicitly cleanup eval objects (prevents OOM and semaphore leaks)
        del test_loader; del test_dataset; del test_episodes; gc.collect()
        return test_acc, test_avg_guesses, test_loss, test_loss_components
    
    def _run_epoch(self, epoch: int):
        # Create rollout dataset
        rollout_episodes = self.simulator.collect_episodes_epoch(
            self.model,
            alpha=self.scheduler.alpha,
            temperature=self.scheduler.temperature,
            argmax=False,
            desc="Collecting Rollout Episodes"
        )
        # Evaluate stats
        active_mask = rollout_episodes["states"]["active_mask"][:, :-1, ...]        # [B, G, *] ignore active mask of final state after last guess
        rollout_wins = (
            rollout_episodes["responses"]["correct"].bool() & active_mask.bool()
        ).any(dim=1).float().sum().item()                                           # [B, G, *] --> [B, *]
        rollout_guesses = active_mask.float().sum().item()                          # [B, G, *] --> scalar
        rollout_games = active_mask[:, 0, ...].float().sum().item()                 # [B, G, *] --> [B, *] --> scalar
        rollout_acc = (rollout_wins / rollout_games)
        rollout_avg_guesses = (rollout_guesses / rollout_games)
        self.logger.log(
            f"[Epoch {epoch}] Rollout Acc: {rollout_acc:.4f} | Rollout Avg Guesses: {rollout_avg_guesses:.4f}"
        )
        
        # Create dataset/loader for processing rollouts
        rollout_dataset = EpisodesDataset(rollout_episodes)
        processing_loader = DataLoader(rollout_dataset, batch_size=self.cfg.processing_batch_size, shuffle=True, num_workers=0)

        # Perform updates over multiple passes
        self.model.train()
        for i in range(self.cfg.rollout_size):
            desc = f"Rollout {i}"
            rollout_loss, rollout_loss_components = self._loop_with_grad(loader=processing_loader, desc=desc)
            self.logger.log(
                f"[Epoch {epoch}][{desc}] Loss: {rollout_loss:.6f} | Actor Loss: {rollout_loss_components['actor']:.6f} | "
                f"Critic Loss: {rollout_loss_components['critic']:.6f} | Entropy Loss: {rollout_loss_components['entropy']:.6f} | "
                f"KL Reg Loss: {rollout_loss_components['kl_reg']:.6f} | KL Guide Loss: {rollout_loss_components['kl_guide']:.6f} | "
                f"KL Best Loss: {rollout_loss_components['kl_best']:.6f}"
            )
        # Explicitly cleanup the rollout objects (prevents OOM and semaphore leaks)
        del processing_loader; del rollout_dataset; del rollout_episodes; gc.collect()

        # Evaluate with argmax policy
        test_acc, test_avg_guesses, test_loss, test_loss_components = self._run_eval(epoch=epoch)
        self.logger.log("\n")  # Newline for readability between epochs
        return test_acc, test_avg_guesses, test_loss, test_loss_components

    def train(self, epochs: int):
        # Setup
        end_epoch = self.last_epoch + epochs
        
        # Show initial alpha, temp
        print(f'Initial alpha: {self.scheduler.alpha:.2f}, Initial temperature: {self.scheduler.temperature:.2f}')

        # First eval and checkpoint at initialization
        if self.last_epoch == 0:
            test_acc, test_avg_guesses, test_loss, test_loss_components = self._run_eval(epoch=self.last_epoch)
            if (self.cfg.checkpoint_dir and self.cfg.save_every):
                self.save_checkpoint(self.last_epoch)

        # Run through epochs
        while self.last_epoch < end_epoch:
            epoch = self.last_epoch + 1

            # Update ref model
            self.ref_model.load_state_dict(self.model.state_dict())

            # Run epoch
            test_acc, test_guesses, test_loss, test_loss_components = self._run_epoch(epoch)

            # Step epoch
            self.scheduler.step_epoch(win_rate=test_acc, avg_guesses=test_guesses, loss=test_loss, loss_components=test_loss_components)

            # Periodic checkpointing
            if (self.cfg.checkpoint_dir and self.cfg.save_every) and (epoch % self.cfg.save_every == 0):
                self.save_checkpoint(epoch)

            # Best model checkpointing
            if self.cfg.checkpoint_dir:
                new_best_model = (
                    test_acc > self.best_accuracy or
                    test_acc == self.best_accuracy and test_guesses < self.best_avg_guesses
                )
                if new_best_model:
                    self.best_accuracy = test_acc
                    self.best_avg_guesses = test_guesses
                    self.best_model.load_state_dict(self.model.state_dict())
                    self.save_checkpoint(epoch)

            # Increment epoch
            self.last_epoch = epoch
