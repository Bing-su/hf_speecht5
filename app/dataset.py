from pathlib import Path

import librosa
import lightning as L
import numpy as np
import pandas as pd
import torch
from munch import Munch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoTokenizer, BatchFeature


class Collator:
    def __init__(self, feature_extractor, tokenizer):
        self.fe = feature_extractor
        self.tk = tokenizer

    def __call__(self, batch):
        audio = [b["audio"] for b in batch]
        text = [b["text"] for b in batch]

        inputs = self.tk(text, padding=True)
        features = self.fe(
            audio_target=audio, sampling_rate=16000, padding=True, return_tensors="pt"
        )

        labels = features["input_values"]
        labels_attention_mask = features["attention_mask"]
        labels = labels.masked_fill(labels_attention_mask.unsqueeze(-1).ne(1), -100)

        length = labels.shape[1]
        if length % 2 == 1:
            labels = labels[:, :-1]
        inputs["labels"] = labels

        if "speaker_id" in batch[0]:
            inputs["speaker_id"] = np.array([b["speaker_id"] for b in batch])

        return BatchFeature(inputs, tensor_type="pt")


class MyDataset(Dataset):
    def __init__(self, cfg: Munch, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.fe = AutoFeatureExtractor.from_pretrained(cfg.model.name_or_path)
        self.tk = AutoTokenizer.from_pretrained(cfg.model.name_or_path)

    def __getitem__(self, idx: int):
        audio_path = self.df.loc[idx, self.cfg.data.audio_col]
        text = self.df.loc[idx, self.cfg.data.text_col]

        audio, _ = librosa.load(audio_path, sr=16000)

        output = {"audio": audio, "text": text}
        if self.cfg.data.speaker_id_col:
            s_id = self.df.loc[idx, self.cfg.data.speaker_id_col]
            output["speaker_id"] = s_id

        return output

    def __len__(self):
        return len(self.df)


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: Munch):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str) -> None:
        data_path = Path(__file__).parent.parent / "kaist-audio-book.csv"
        df = pd.read_csv(data_path)
        train_df, val_df = train_test_split(
            df, test_size=0.05, random_state=42, stratify=df["speaker_id"]
        )
        self.train = MyDataset(self.cfg, train_df)
        self.val = MyDataset(self.cfg, val_df)
        self.collator = Collator(self.train.fe, self.train.tk)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            collate_fn=self.collator,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.cfg.data.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            collate_fn=self.collator,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.cfg.data.num_workers > 0,
        )
