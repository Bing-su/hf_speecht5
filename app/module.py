from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from loguru import logger
from munch import Munch
from pytorch_optimizer import create_optimizer
from transformers import SpeechT5Config, SpeechT5ForTextToSpeech


class MyModule(L.LightningModule):
    def __init__(self, cfg: Munch):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        config = SpeechT5Config.from_pretrained(cfg.model.name_or_path)
        self.model = SpeechT5ForTextToSpeech(config)
        self.speaker_embedding = None
        if self.cfg.model.num_speakers > 1:
            self.speaker_embedding = torch.nn.Embedding(
                self.cfg.model.num_speakers, self.model.config.speaker_embedding_dim
            )

        if self.cfg.model.get("resume") and Path(self.cfg.model.resume).exists():
            logger.info(f"load state dict from {self.cfg.model.resume}")
            sd = torch.load(self.cfg.model.resume, map_location="cpu")
            self.load_state_dict(sd["state_dict"])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self, batch):
        speaker_id = batch.pop("speaker_id", None)
        if self.speaker_embedding is not None and speaker_id is not None:
            batch["speaker_embeddings"] = self.speaker_embedding(speaker_id)
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        output = self.step(batch)
        loss = output.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        loss = output.loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer(
            self,
            "madgrad",
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            decouple_decay=True,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.train.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler_config = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler_config]

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.speaker_embedding is not None:
            self.model.config.num_speakers = self.cfg.model.num_speakers
            torch.save(
                self.speaker_embedding.state_dict(), path / "speaker_embedding.pt"
            )

        self.model.save_pretrained(path)
