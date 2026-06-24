# General
from typing import Union

# Torch

# Wordle
from wordle.utils import Config, to_float



class SchedulerConfig(Config):
    def __init__(
            self,
            init_alpha: float = 0.99,
            min_alpha: float = 0.00,
            alpha_step: float = 0.05,
            init_temperature: float = 5.00,
            min_temperature: float = 0.25,
            temperature_decay: float = 0.90,
            patience: int = 1,
            warmup_steps: int = 1000,
        ):
        self.init_alpha = init_alpha
        self.min_alpha = min_alpha
        self.alpha_step = alpha_step
        self.init_temperature = init_temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.patience = patience
        self.warmup_steps = warmup_steps


class Scheduler:
    def __init__(self, opt, cfg: SchedulerConfig):
        # Main objects
        self.opt = opt
        self.cfg = cfg
        # Warmup
        self.warmup_steps = self.cfg.warmup_steps
        self.warmup_factor = 1.0 / self.warmup_steps
        # Greedification
        self.init_alpha = self.cfg.init_alpha
        self.min_alpha = self.cfg.min_alpha
        self.alpha_step = self.cfg.alpha_step
        self.init_temperature = self.cfg.init_temperature
        self.min_temperature = self.cfg.min_temperature
        self.temperature_decay = self.cfg.temperature_decay
        self.patience = self.cfg.patience
        # State
        self.step_idx = 0
        self.steps_since_improvement = 0
        self.alpha = self.init_alpha
        self.temperature = self.init_temperature
        self.best_metrics = {
            "search_kl": float('inf'),
            "win_rate": 0.0,
            "avg_guesses": float('inf')
        }
        # Set initial LR
        for param_group in self.opt.param_groups:
            param_group["base_lr"] = param_group["lr"]
        if self.warmup_steps > 0:
            self._scale_lrs(1.0 / self.warmup_steps)
    
    def _show(self):
        state = self.state_dict()
        print(f"Scheduler State:")
        line = ""
        for key, value in state.items():
            line += f"{key}: {value} | "
        print(line)
    
    def _set_base_lrs(self, base_lrs: Union[float, list[float]]):
        if isinstance(base_lrs, float):
            base_lrs = [base_lrs] * len(self.opt.param_groups)
        for param_group, base_lr in zip(self.opt.param_groups, base_lrs):
            param_group["base_lr"] = base_lr

    def _scale_lrs(self, scale):
        for param_group in self.opt.param_groups:
            param_group["lr"] = param_group["base_lr"] * scale

    def step(self):
        self.step_idx += 1
        # No warmup
        if self.warmup_steps <= 0:
            return
        # LR warmup
        next_step = self.step_idx + 1
        if next_step <= self.warmup_steps:
            self._scale_lrs(next_step / self.warmup_steps)
        else:
            self._scale_lrs(1.0)

    def step_epoch(self, win_rate: float, avg_guesses: float, loss: float, loss_components: dict):
        self.step_idx += 1
        no_improvement = True
        # Track general improvement in any area
        if win_rate > self.best_metrics["win_rate"]:
            self.best_metrics["win_rate"] = to_float(win_rate)
            no_improvement = False
        if avg_guesses < self.best_metrics["avg_guesses"]:
            self.best_metrics["avg_guesses"] = to_float(avg_guesses)
            no_improvement = False
        if loss_components["search_kl"] < self.best_metrics["search_kl"]:
            self.best_metrics["search_kl"] = to_float(loss_components["search_kl"])
            no_improvement = False

        # Update patience counter
        if no_improvement:
            self.steps_since_improvement += 1
        else:
            self.steps_since_improvement = 0

        # Greedify after plateau
        if self.steps_since_improvement >= self.patience:
            c1 = (self.alpha > self.min_alpha)
            c2 = (self.temperature > self.min_temperature)
            if (c1 or c2):
                print(f'No improvement for {self.steps_since_improvement} steps. Greedifying...')
                if c1:
                    new_alpha = max(self.alpha - self.alpha_step, self.min_alpha)
                    print(f'  -> Evolved alpha: {self.alpha:.2f} -> {new_alpha:.2f}, temp: {self.temperature:.2f} -> {self.temperature:.2f}')
                    self.alpha = new_alpha
                elif c2:
                    new_temperature = max(self.temperature * self.temperature_decay, self.min_temperature)
                    print(f'  -> Evolved temperature: {self.temperature:.2f} -> {new_temperature:.2f}, alpha: {self.alpha:.2f} -> {self.alpha:.2f}')
                    self.temperature = new_temperature
                self.steps_since_improvement = 0
                self.best_metrics["search_kl"] = float('inf')

    def state_dict(self):
        return {
            "step_idx": self.step_idx,
            "steps_since_improvement": self.steps_since_improvement,
            "alpha": self.alpha,
            "temperature": self.temperature,
            "best_metrics": self.best_metrics,
        }

    def load_state_dict(self, state):
        self.step_idx = state["step_idx"]
        self.steps_since_improvement = state["steps_since_improvement"]
        self.alpha = state["alpha"]
        self.temperature = state["temperature"]
        self.best_metrics = state["best_metrics"]
