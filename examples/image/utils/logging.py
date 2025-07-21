import logging
import os
from logging import Logger
from pathlib import Path
from typing import Optional

import torch
import wandb


def get_logger(log_path: str, rank: int) -> Logger:
    if rank != 0:
        return logging.getLogger("dummy")

    logger = logging.getLogger()
    default_level = logging.INFO

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(default_level)

    formatter = logging.Formatter(
        "%(levelname)s | %(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    info_file_handler = logging.FileHandler(log_path, mode="a")
    info_file_handler.setLevel(default_level)
    info_file_handler.setFormatter(formatter)
    logger.addHandler(info_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(default_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class TrainLogger:
    def __init__(
        self,
        log_dir: Path,
        rank: int,
        enable_wandb: bool = False,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
    ) -> None:
        self.log_dir = log_dir
        self.enable_wandb = enable_wandb and (rank == 0)
        self.project = project
        self.entity = entity
        self.group = group

        self._init_text_logger(rank=rank)
        if self.enable_wandb:
            self._init_wandb()

    def _init_text_logger(self, rank: int) -> None:
        log_path = self.log_dir / "log.txt"
        self._logger = get_logger(log_path=str(log_path), rank=rank)

    def _init_wandb(self) -> None:
        wandb_run_id_path = self.log_dir / "wandb_run.id"

        try:
            wandb_run_id = wandb_run_id_path.read_text()
        except FileNotFoundError:
            wandb_run_id = wandb.util.generate_id()
            wandb_run_id_path.write_text(wandb_run_id)

        self.wandb_logger = wandb.init(
            id=wandb_run_id,
            project=self.project,
            group=self.group,
            dir=self.log_dir,
            entity=self.entity,
            resume="allow",
        )

    def log_metric(self, value: float, name: str, stage: str, step: int) -> None:
        self._logger.info(f"[{step}] {stage} {name}: {value:.3f}")
        if self.enable_wandb:
            self.wandb_logger.log({f"{stage}/{name}": value}, step=step)

    def log_lr(self, value: float, step: int) -> None:
        if self.enable_wandb:
            self.wandb_logger.log({"Optimization/LR": value}, step=step)

    def info(self, msg: str, step: Optional[int] = None) -> None:
        step_str = f"[{step}] " if step is not None else ""
        self._logger.info(f"{step_str}{msg}")

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def finish(self) -> None:
        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
        if self.enable_wandb:
            wandb.finish()

    @staticmethod
    def log_devices(device: torch.device, logger: Logger) -> None:
        if device.type == "cuda":
            logger.info(f"Found {torch.cuda.device_count()} CUDA devices.")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(
                    f"{props.name}\t Memory: {props.total_memory / (1024**3):.2f}GB"
                )
        else:
            logger.warning(f"WARNING: Using device {device}")
        logger.info(f"Found {os.cpu_count()} total number of CPUs.")
