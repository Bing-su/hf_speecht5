import lightning as L
import torch
import numpy as np
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from munch import Munch
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
        features = self.fe(audio_target=audio, sampling_rate=16000, padding=True)
        inputs["labels"] = features["input_values"]

        attention_mask = features.get("attention_mask")
        if attention_mask is not None:
            inputs["decoder_attention_mask"] = attention_mask

        if "speaker_id" in batch[0]:
            inputs["speaker_id"] = np.array([b["speaker_id"] for b in batch])
        return BatchFeature(inputs, tensor_type="pt")


class MyDataset(Dataset):
    def __init__(self, cfg: Munch, dataset: HFDataset):
        self.cfg = cfg
        self.dataset = dataset
        self.fe = AutoFeatureExtractor.from_pretrained(cfg.model.name_or_path)
        self.tk = AutoTokenizer.from_pretrained(cfg.model.name_or_path)

    def __getitem__(self, idx: int):
        audio = self.dataset[idx][self.cfg.data.audio_col]["array"]
        text = self.dataset[idx][self.cfg.data.text_col]

        output = {"audio": audio, "text": text}
        if self.cfg.data.speaker_id_col:
            s_id = self.dataset[idx][self.cfg.data.speaker_id_col]
            output["speaker_id"] = s_id

        return output

    def __len__(self):
        return len(self.dataset)


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: Munch):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str) -> None:
        ds = load_from_disk(self.cfg.data.path)
        dss = ds.train_test_split(test_size=0.05, seed=42, shuffle=True)
        self.train = MyDataset(self.cfg, dss["train"])
        self.val = MyDataset(self.cfg, dss["test"])
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
