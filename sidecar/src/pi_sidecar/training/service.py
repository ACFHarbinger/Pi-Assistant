"""
Training Service.

Manages training runs with start/stop/status capabilities.
Exposes training pipeline to IPC for agent-initiated training.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["TrainingService", "RunStatus", "RunInfo"]


class RunStatus(Enum):
    """Status of a training run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunInfo:
    """Information about a training run."""
    run_id: str
    status: RunStatus
    config: dict[str, Any]
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    model_path: str | None = None


class TrainingService:
    """
    Service for managing training runs.
    
    Provides async interface for:
    - Starting new training runs
    - Stopping running jobs
    - Querying run status
    - Listing historical runs
    """
    
    def __init__(self, output_dir: str = "~/.pi-assistant/models") -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Active runs
        self._runs: dict[str, RunInfo] = {}
        self._tasks: dict[str, asyncio.Task] = {}  # type: ignore
        
        # Load run history from disk
        self._load_history()
    
    def _load_history(self) -> None:
        """Load historical run info from disk."""
        # Check for runs.json in output dir
        history_file = self.output_dir / "runs.json"
        if history_file.exists():
            try:
                import json
                with open(history_file) as f:
                    data = json.load(f)
                for run_data in data.get("runs", []):
                    self._runs[run_data["run_id"]] = RunInfo(
                        run_id=run_data["run_id"],
                        status=RunStatus(run_data["status"]),
                        config=run_data.get("config", {}),
                        started_at=datetime.fromisoformat(run_data["started_at"]) if run_data.get("started_at") else None,
                        completed_at=datetime.fromisoformat(run_data["completed_at"]) if run_data.get("completed_at") else None,
                        metrics=run_data.get("metrics", {}),
                        error=run_data.get("error"),
                        model_path=run_data.get("model_path"),
                    )
            except Exception as e:
                logger.warning(f"Failed to load run history: {e}")
    
    def _save_history(self) -> None:
        """Persist run history to disk."""
        import json
        
        history_file = self.output_dir / "runs.json"
        data = {
            "runs": [
                {
                    "run_id": run.run_id,
                    "status": run.status.value,
                    "config": run.config,
                    "started_at": run.started_at.isoformat() if run.started_at else None,
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                    "metrics": run.metrics,
                    "error": run.error,
                    "model_path": run.model_path,
                }
                for run in self._runs.values()
            ]
        }
        
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)
    
    async def start(self, config: dict[str, Any]) -> str:
        """
        Start a new training run.
        
        Args:
            config: Training configuration dict with:
                - backbone: Name of backbone to use
                - head: Name of head to use
                - backbone_config: Dict of backbone parameters
                - head_config: Dict of head parameters
                - training: Training hyperparameters
                - data: Dataset configuration
        
        Returns:
            run_id: Unique identifier for this run
        """
        run_id = str(uuid.uuid4())[:8]
        
        run_info = RunInfo(
            run_id=run_id,
            status=RunStatus.PENDING,
            config=config,
            started_at=datetime.now(),
        )
        self._runs[run_id] = run_info
        
        # Spawn training task
        task = asyncio.create_task(self._run_training(run_id, config))
        self._tasks[run_id] = task
        
        logger.info(f"Started training run: {run_id}")
        return run_id
    
    async def _run_training(self, run_id: str, config: dict[str, Any]) -> None:
        """Execute training in background."""
        run_info = self._runs[run_id]
        run_info.status = RunStatus.RUNNING
        
        try:
            # Import here to avoid circular deps and lazy load heavy modules
            from pi_sidecar.ml.models.composed import build_model
            from pi_sidecar.ml.pipeline.accelerated import AcceleratedTrainer, AcceleratedTrainerConfig
            
            # Build model
            model = build_model(
                backbone_name=config.get("backbone", "transformer"),
                head_name=config.get("head", "classification"),
                backbone_config=config.get("backbone_config", {}),
                head_config=config.get("head_config", {}),
            )
            
            # Create dummy data for now (would be replaced by actual data loading)
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            
            data_config = config.get("data", {})
            batch_size = data_config.get("batch_size", 32)
            num_samples = data_config.get("num_samples", 1000)
            seq_len = data_config.get("seq_len", 50)
            input_dim = config.get("backbone_config", {}).get("input_dim", 10)
            
            # Generate synthetic data
            x = torch.randn(num_samples, seq_len, input_dim)
            y = torch.randint(0, config.get("head_config", {}).get("num_classes", 10), (num_samples,))
            
            dataset = TensorDataset(x, y)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Configure trainer
            training_config = config.get("training", {})
            trainer_config = AcceleratedTrainerConfig(
                max_epochs=training_config.get("max_epochs", 10),
                learning_rate=training_config.get("learning_rate", 3e-4),
                batch_size=batch_size,
                output_dir=str(self.output_dir),
                run_name=run_id,
                mixed_precision=training_config.get("mixed_precision", "no"),
            )
            
            # Train
            trainer = AcceleratedTrainer(
                model=model,
                train_loader=train_loader,
                config=trainer_config,
                loss_fn=torch.nn.CrossEntropyLoss(),
            )
            
            result = trainer.train()
            
            # Update run info
            run_info.status = RunStatus.COMPLETED
            run_info.completed_at = datetime.now()
            run_info.metrics = result
            run_info.model_path = str(self.output_dir / run_id / "best_model.pt")
            
            logger.info(f"Training completed: {run_id}")
            
        except asyncio.CancelledError:
            run_info.status = RunStatus.CANCELLED
            run_info.completed_at = datetime.now()
            logger.info(f"Training cancelled: {run_id}")
            
        except Exception as e:
            run_info.status = RunStatus.FAILED
            run_info.completed_at = datetime.now()
            run_info.error = str(e)
            logger.error(f"Training failed: {run_id} - {e}")
        
        finally:
            self._save_history()
            if run_id in self._tasks:
                del self._tasks[run_id]
    
    async def stop(self, run_id: str) -> bool:
        """
        Stop a running training job.
        
        Args:
            run_id: ID of the run to stop
        
        Returns:
            True if stopped, False if not found or not running
        """
        if run_id not in self._tasks:
            return False
        
        task = self._tasks[run_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        return True
    
    async def status(self, run_id: str) -> dict[str, Any]:
        """
        Get status of a training run.
        
        Args:
            run_id: ID of the run
        
        Returns:
            Dict with run status and metrics
        """
        if run_id not in self._runs:
            return {"error": f"Run not found: {run_id}"}
        
        run = self._runs[run_id]
        return {
            "run_id": run.run_id,
            "status": run.status.value,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "metrics": run.metrics,
            "error": run.error,
            "model_path": run.model_path,
        }
    
    async def list_runs(self) -> list[dict[str, Any]]:
        """
        List all training runs.
        
        Returns:
            List of run status dicts
        """
        return [
            await self.status(run_id)
            for run_id in sorted(self._runs.keys(), reverse=True)
        ]
