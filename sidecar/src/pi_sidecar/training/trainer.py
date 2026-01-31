"""Training orchestration with progress streaming."""

from typing import Optional, Callable, Any
from pathlib import Path
import asyncio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import torch

from .lightning_module import PiLightningModule
from .data_module import PiDataModule


class ProgressCallback(Callback):
    """Callback for streaming training progress."""

    def __init__(self, progress_fn: Optional[Callable[[dict], None]] = None):
        self.progress_fn = progress_fn
        self.current_epoch = 0
        self.total_epochs = 0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.total_epochs = trainer.max_epochs or 0

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.current_epoch = trainer.current_epoch
        self._emit_progress("epoch_start", {"epoch": self.current_epoch})

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx % 10 == 0:  # Report every 10 batches
            loss = outputs["loss"].item() if isinstance(outputs, dict) else outputs.item()
            self._emit_progress("batch", {
                "epoch": self.current_epoch,
                "batch": batch_idx,
                "loss": loss,
            })

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in trainer.callback_metrics.items()}
        self._emit_progress("epoch_end", {
            "epoch": self.current_epoch,
            "metrics": metrics,
        })

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._emit_progress("complete", {"epochs": self.current_epoch + 1})

    def _emit_progress(self, event: str, data: dict) -> None:
        if self.progress_fn:
            self.progress_fn({"event": event, **data})


class TrainingOrchestrator:
    """Orchestrates model training with progress streaming."""

    def __init__(
        self,
        model_name: str = "gpt2",
        output_dir: Optional[Path] = None,
        progress_fn: Optional[Callable[[dict], None]] = None,
    ):
        self.model_name = model_name
        self.output_dir = output_dir or Path("./checkpoints")
        self.progress_fn = progress_fn
        
        self.module: Optional[PiLightningModule] = None
        self.trainer: Optional[pl.Trainer] = None

    def prepare(
        self,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
    ) -> None:
        """Prepare the model for training."""
        self.module = PiLightningModule(
            model_name=self.model_name,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
        )

    def train(
        self,
        train_texts: list[str],
        val_texts: Optional[list[str]] = None,
        epochs: int = 3,
        batch_size: int = 8,
        accelerator: str = "auto",
    ) -> dict:
        """Run training."""
        if self.module is None:
            self.prepare()
        
        # Create data module
        data_module = PiDataModule(
            tokenizer=self.module.tokenizer,
            train_texts=train_texts,
            val_texts=val_texts or [],
            batch_size=batch_size,
        )
        
        # Callbacks
        callbacks = [
            ProgressCallback(self.progress_fn),
            ModelCheckpoint(
                dirpath=self.output_dir,
                filename="model-{epoch:02d}-{val_loss:.2f}",
                save_top_k=2,
                monitor="val_loss" if val_texts else "train_loss",
                mode="min",
            ),
        ]
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            callbacks=callbacks,
            enable_progress_bar=False,  # We use our own progress
            logger=False,
        )
        
        # Train
        self.trainer.fit(self.module, data_module)
        
        return {
            "epochs": epochs,
            "final_loss": self.trainer.callback_metrics.get("train_loss", 0),
            "checkpoint_dir": str(self.output_dir),
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the trained model."""
        if self.module is None:
            raise RuntimeError("No model to save")
        
        save_path = path or self.output_dir / "final_model"
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.module.model.save_pretrained(save_path)
        self.module.tokenizer.save_pretrained(save_path)
        
        return save_path
