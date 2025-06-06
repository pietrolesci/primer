# from typing import Any
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.model_helpers import is_overridden
from torch import Tensor
from torch.optim import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.aggregation import SumMetric

from primer.utilities import get_logger

logger = get_logger("grad_accum")


class GradientAccumulationScheduler(Callback):
    """Allows to change the number of accumulated batches during training.

    This is similar to lightning.callbacks.GradientAccumulationScheduler but based on steps rather than epochs.
    """

    PREFIX = "optim"

    def __init__(self, scheduling: dict[int, int] | None = None) -> None:
        super().__init__()
        scheduling = scheduling or {0: 1}
        minimal_step = min(scheduling.keys())
        if minimal_step < 0:
            raise IndexError(f"Step indexing from 1, step {minimal_step} cannot be interpreted correct")
        if minimal_step != 0:  # if user didn't define first step accumulation factor
            scheduling.update({0: 1})

        self.scheduling = scheduling
        self.steps = sorted(scheduling.keys())

        # self.counter = {"num_instances": 0, "num_batches": 0, "num_tokens": 0}
        self.counter = MetricCollection(
            {"num_batches": SumMetric(), "num_instances": SumMetric(), "num_million_tokens": SumMetric()},
            prefix=f"{self.PREFIX}/",
        )

    def going_to_accumulate_grad_batches(self) -> bool:
        return any(v > 1 for v in self.scheduling.values())

    def get_accumulate_grad_batches(self, step: int) -> int:
        accumulate_grad_batches = 1
        for iter_step in reversed(self.steps):
            if step >= iter_step:
                accumulate_grad_batches = self.scheduling[iter_step]
                break
        return accumulate_grad_batches

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Performns a configuration validation before training starts and raises errors for incompatible settings."""

        if not pl_module.automatic_optimization:
            raise RuntimeError(
                """Automatic gradient accumulation and the `GradientAccumulationScheduler` is not supported for
                manual optimization. Please remove the callback or switch to automatic optimization."""
            )

        overridden_optimizer_step = is_overridden("optimizer_step", pl_module)
        overridden_optimizer_zero_grad = is_overridden("optimizer_zero_grad", pl_module)
        going_to_accumulate_grad_batches = self.going_to_accumulate_grad_batches()
        has_overridden_optimization_functions = overridden_optimizer_step or overridden_optimizer_zero_grad
        if has_overridden_optimization_functions and going_to_accumulate_grad_batches:
            logger.warning(
                "When using `Trainer(accumulate_grad_batches != 1)` and overriding"
                " `LightningModule.optimizer_{step,zero_grad}`, the hooks will not be called on every batch"
                " (rather, they are called on every optimization step)."
            )

        # local import to avoid circular import
        from lightning.pytorch.strategies import DeepSpeedStrategy

        if isinstance(trainer.strategy, DeepSpeedStrategy):
            raise RuntimeError(
                f"The `{type(trainer.strategy).__name__}` does not support `accumulate_grad_batches` changing"
                " between steps."
            )
        if trainer.accumulate_grad_batches != 1:
            raise ValueError(
                "You have set `accumulate_grad_batches` and are using the `GradientAccumulationScheduler`"
                " callback. Either remove `accumulate_grad_batches` from the Trainer or remove the callback."
            )

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Tensor | dict[str, Tensor], batch_idx: int
    ) -> None:
        batch = batch["input_ids"] if isinstance(batch, dict) else batch

        # Update metrics using the MetricCollection
        self.counter["num_batches"].update(1)
        self.counter["num_instances"].update(batch.shape[0])
        self.counter["num_million_tokens"].update(batch.numel() / 1e6)

        # Maybe change the number of accumulated batches
        prev_accumulate_grad_batches = trainer.accumulate_grad_batches
        trainer.accumulate_grad_batches = self.get_accumulate_grad_batches(trainer.global_step)
        if trainer.accumulate_grad_batches != prev_accumulate_grad_batches:
            eff_bs = batch.shape[0] * trainer.accumulate_grad_batches * trainer.num_devices * trainer.num_nodes
            logger.info(
                f"Number of accumulation batches changed from {prev_accumulate_grad_batches} "
                f"to {trainer.accumulate_grad_batches} at step {trainer.global_step}. "
                f"Effective batch size = {eff_bs}."
            )

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer) -> None:
        counts = self.counter.compute()
        counts = {f"{k}_per_step": v for k, v in counts.items()}
        pl_module.log_dict(
            counts,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,  # Synchronize across distributed processes is done by torchmetrics internally
        )

        # Reset metrics after logging
        self.counter.reset()
