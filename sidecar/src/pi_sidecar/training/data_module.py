"""PyTorch Lightning DataModule for training data."""

from typing import Optional, List
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer


class TextDataset(Dataset):
    """Simple text dataset for language model fine-tuning."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize the dataset.
        Args:
            self: The dataset.
            texts: The texts.
            tokenizer: The tokenizer.
            max_length: The maximum length of the texts.
        Returns:
            None
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        Args:
            self: The dataset.
        Returns:
            int: The length of the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Get an item from the dataset.
        Args:
            self: The dataset.
            idx: The index of the item.
        Returns:
            dict: The item.
        """
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


class PiDataModule(pl.LightningDataModule):
    """Lightning DataModule for training data loading."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        train_texts: Optional[List[str]] = None,
        val_texts: Optional[List[str]] = None,
        data_path: Optional[Path] = None,
        max_length: int = 512,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        """
        Initialize the data module.
        Args:
            self: The data module.
            tokenizer: The tokenizer.
            train_texts: The training texts.
            val_texts: The validation texts.
            data_path: The path to the data file.
            max_length: The maximum length of the texts.
            batch_size: The batch size.
            num_workers: The number of workers.
        Returns:
            None
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.train_texts = train_texts or []
        self.val_texts = val_texts or []
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataset: Optional[TextDataset] = None
        self.val_dataset: Optional[TextDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets.
        Args:
            self: The data module.
            stage: The stage to set up.
        Returns:
            None
        """
        # Load from file if path provided
        if self.data_path and self.data_path.exists():
            with open(self.data_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            split_idx = int(len(lines) * 0.9)
            self.train_texts = lines[:split_idx]
            self.val_texts = lines[split_idx:]
        
        if stage == "fit" or stage is None:
            if self.train_texts:
                self.train_dataset = TextDataset(
                    self.train_texts,
                    self.tokenizer,
                    self.max_length,
                )
            if self.val_texts:
                self.val_dataset = TextDataset(
                    self.val_texts,
                    self.tokenizer,
                    self.max_length,
                )

    def train_dataloader(self) -> Optional[DataLoader]:
        """
        Create training dataloader.
        Args:
            self: The data module.
        Returns:
            Optional[DataLoader]: The training dataloader.
        """
        if self.train_dataset is None:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Create validation dataloader.
        Args:
            self: The data module.
        Returns:
            Optional[DataLoader]: The validation dataloader.
        """
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
