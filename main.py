import time
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from munch import Munch
from rich.traceback import install

from app.dataset import DataModule
from app.module import MyModule

install(show_locals=True)
torch.set_float32_matmul_precision("medium")


class Timer:
    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self):
        self.timer = time.time()

    def __exit__(self, *args, **kwargs):
        logger.info(f"elapsed time on {self.name}: {time.time() - self.timer:.2f}s")


def train():
    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open(encoding="utf-8") as file:
        cfg: Munch = Munch.fromYAML(file)

    logger.info("set trainer")
    checkpoints = ModelCheckpoint(
        "checkpoints", monitor="val_loss", mode="min", save_last=True
    )
    callbacks = [checkpoints, LearningRateMonitor(), RichProgressBar()]

    seed = cfg.train.get("seed")
    seed = L.seed_everything(seed)
    cfg.train.seed = seed
    logger.info(f"seed: {seed}")

    now = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    wandb_logger = WandbLogger(name=now, project="speecht5_tts", config=cfg.toDict())

    trainer = L.Trainer(
        precision=cfg.train.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        fast_dev_run=cfg.train.get("fast_dev_run", False),
        max_epochs=cfg.train.max_epochs,
        gradient_clip_val=cfg.train.get("gradient_clip_val", None),
    )

    resume_from_checkpoint = cfg.train.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        logger.info(f"resume from {resume_from_checkpoint!r}")

    logger.info("create model")
    with Timer("model"):
        module = MyModule(cfg)

    logger.info("create dataset")
    with Timer("dataset"):
        datamodule = DataModule(cfg)

    logger.info("training start")
    trainer.fit(model=module, datamodule=datamodule, ckpt_path=resume_from_checkpoint)

    best_model_path = Path(checkpoints.best_model_path)
    if best_model_path.exists() and best_model_path.is_file():
        logger.info(f"best model: {best_model_path}")
        best_model = MyModule.load_from_checkpoint(best_model_path)
        best_model.save("save")


if __name__ == "__main__":
    train()
