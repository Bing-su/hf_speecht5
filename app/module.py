from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from munch import Munch
from pytorch_optimizer import create_optimizer
from transformers import SpeechT5ForTextToSpeech


class MyModule(L.LightningModule):
    def __init__(self, cfg: Munch):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.model = SpeechT5ForTextToSpeech.from_pretrained(cfg.model.name_or_path)
        self.speaker_embedding = None
        if self.cfg.model.num_speakers > 1:
            self.speaker_embedding = torch.nn.Embedding(
                self.cfg.model.num_speakers, self.model.config.speaker_embedding_dim
            )

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
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer(
            self,
            "adafactor",
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            warmup_init=True,
        )

        return optimizer

    def save(self, path: str | Path):
        path = Path(path)
        if self.speaker_embedding is not None:
            self.model.config.num_speakers = self.cfg.model.num_speakers
            torch.save(
                self.speaker_embedding.state_dict(), path / "speaker_embedding.pt"
            )

        self.model.save_pretrained(path)
