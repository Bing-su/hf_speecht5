import lightning as L
import torch
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
        inputs = self.tk.pad([b["tokens"] for b in batch], padding=True)
        target = self.fe.pad([b["target"] for b in batch], padding=True)
        inputs["labels"] = target["input_values"]
        inputs["stop_labels"] = target["stop_labels"]
        decoder_attention_mask = target.get("decoder_attention_mask")
        if decoder_attention_mask:
            inputs["decoder_attention_mask"] = decoder_attention_mask
        return BatchFeature(inputs, tensor_type="pt")


class MyDataset(Dataset):
    def __init__(self, cfg: Munch, dataset: HFDataset):
        self.cfg = cfg
        self.dataset = dataset
        self.fe = AutoFeatureExtractor.from_pretrained(cfg.model.name_or_path)
        self.tk = AutoTokenizer.from_pretrained(cfg.model.name_or_path)

    def __getitem__(self, idx: int):
        audio = self.dataset[idx][self.cfg.data.audio_col]
        text = self.dataset[idx][self.cfg.data.text_col]

        tokens = self.tk(text)
        target = self.fe(audio_target=audio, sampling_rate=16000)

        return {"tokens": tokens, "target": target}

    def __len__(self):
        return len(self.dataset)


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: Munch):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str) -> None:
        ds = load_from_disk(self.cfg.data.path)
        dss = ds["train"].train_test_split(
            test_size=0.05, seed=42, shuffle=True, stratify_by_column="speaker_id"
        )
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
