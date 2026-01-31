"""
Accelerated Trainer.

Multi-GPU training with Hugging Face Accelerate and optional DeepSpeed integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    Accelerator = None  # type: ignore

from pi_sidecar.ml.pipeline.base import BaseCallback, BaseTrainer

logger = logging.getLogger(__name__)

__all__ = ["AcceleratedTrainerConfig", "AcceleratedTrainer"]


@dataclass
class AcceleratedTrainerConfig:
    """Configuration for accelerated training."""
    # Training params
    max_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Accelerate params
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    gradient_checkpointing: bool = False
    
    # Logging/Checkpointing
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 1
    output_dir: str = "~/.pi-assistant/models"
    run_name: str = "run"
    
    # Distributed
    deepspeed_config: str | None = None  # Path to DeepSpeed config
    
    # Extra
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


class AcceleratedTrainer(BaseTrainer):
    """
    Multi-GPU trainer using Hugging Face Accelerate.
    
    Supports:
    - Multi-GPU (DDP)
    - Mixed precision (FP16/BF16)
    - Gradient accumulation
    - Optional DeepSpeed integration
    - Automatic checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: AcceleratedTrainerConfig | None = None,
        loss_fn: Callable | None = None,
        callbacks: list[BaseCallback] | None = None,
    ) -> None:
        if not HAS_ACCELERATE:
            raise ImportError(
                "Accelerate is required for multi-GPU training. "
                "Install with: pip install accelerate"
            )
        
        self.config = config or AcceleratedTrainerConfig()
        self.callbacks = callbacks or []
        
        # Set seed for reproducibility
        set_seed(self.config.seed)
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=str(Path(self.config.output_dir).expanduser()),
        )
        
        # Setup optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Setup scheduler
        num_training_steps = len(train_loader) * self.config.max_epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=num_training_steps,
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, self.train_loader, self.scheduler = (
            self.accelerator.prepare(model, optimizer, train_loader, scheduler)
        )
        
        self.val_loader = (
            self.accelerator.prepare(val_loader) if val_loader else None
        )
        
        # Loss function
        self.loss_fn = loss_fn or nn.MSELoss()
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir).expanduser() / self.config.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")
    
    def train(self) -> dict[str, Any]:
        """Run the full training loop."""
        # Notify callbacks
        for cb in self.callbacks:
            cb.on_train_begin()
        
        self.accelerator.init_trackers(self.config.run_name)
        
        for epoch in range(self.config.max_epochs):
            # Epoch callbacks
            for cb in self.callbacks:
                cb.on_epoch_begin(epoch)
            
            # Train
            train_metrics = self._train_epoch(epoch)
            
            # Validate
            val_metrics = {}
            if self.val_loader:
                val_metrics = self._validate_epoch(epoch)
            
            # Merge metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Epoch callbacks
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, metrics)
            
            # Checkpointing
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, metrics)
            
            # Log
            if self.accelerator.is_main_process:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.max_epochs} - "
                    f"Train Loss: {metrics.get('train_loss', 0):.4f} - "
                    f"Val Loss: {metrics.get('val_loss', 'N/A')}"
                )
        
        # End training
        for cb in self.callbacks:
            cb.on_train_end()
        
        self.accelerator.end_training()
        
        return {
            "final_train_loss": train_metrics.get("train_loss"),
            "final_val_loss": val_metrics.get("val_loss"),
            "best_val_loss": self.best_val_loss,
            "total_steps": self.global_step,
        }
    
    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Batch callbacks
            for cb in self.callbacks:
                cb.on_batch_begin(batch_idx)
            
            with self.accelerator.accumulate(self.model):
                # Forward
                if isinstance(batch, dict):
                    x = batch.get("input", batch.get("x"))
                    y = batch.get("target", batch.get("y"))
                elif isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch, None
                
                output = self.model(x)
                
                if y is not None:
                    loss = self.loss_fn(output, y)
                else:
                    # Assume model returns loss directly
                    loss = output if isinstance(output, torch.Tensor) and output.dim() == 0 else output.mean()
                
                # Backward
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_every_n_steps == 0:
                self.accelerator.log(
                    {"train_loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]},
                    step=self.global_step,
                )
            
            # Batch callbacks
            for cb in self.callbacks:
                cb.on_batch_end(batch_idx, {"loss": loss.item()})
        
        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}
    
    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            if isinstance(batch, dict):
                x = batch.get("input", batch.get("x"))
                y = batch.get("target", batch.get("y"))
            elif isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch, None
            
            output = self.model(x)
            
            if y is not None:
                loss = self.loss_fn(output, y)
            else:
                loss = output if isinstance(output, torch.Tensor) and output.dim() == 0 else output.mean()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Track best
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self._save_checkpoint(epoch, {"val_loss": avg_loss}, is_best=True)
        
        self.accelerator.log({"val_loss": avg_loss}, step=self.global_step)
        
        return {"val_loss": avg_loss}
    
    def _save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint."""
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            # Unwrap model for saving
            unwrapped = self.accelerator.unwrap_model(self.model)
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": unwrapped.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": metrics,
                "config": self.config.__dict__,
            }
            
            # Save regular checkpoint
            path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint: {path}")
            
            # Save best model
            if is_best:
                best_path = self.output_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best model: {best_path}")
