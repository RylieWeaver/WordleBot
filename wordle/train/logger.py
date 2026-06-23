# General
import time
from pathlib import Path

# Torch

# Wordle
from wordle.utils import Config



class LoggerConfig(Config):
    def __init__(
            self,
            log_dir=None,
            prefix="",
            affix="",
        ):
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "log"
        self.prefix = prefix
        self.affix = affix

class Logger:
    def __init__(self, cfg: LoggerConfig):
        self.cfg = cfg
        self.log_dir = self.cfg.log_dir
        self.affix = self.cfg.affix
        self.prefix = self.cfg.prefix
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_time(self, message: str = "") -> None:
        print(message)
        log_path = self.log_dir / f"{self.prefix}timestamp{self.affix}.log"
        with log_path.open("a") as f:
            f.write(f"{message} Timestamp: {time.time()}\n")

    def log(self, message: str) -> None:
        print(message)
        log_path = self.log_dir / f"{self.prefix}training{self.affix}.log"
        with log_path.open("a") as f:
            f.write(message + "\n")
