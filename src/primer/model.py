import importlib.util
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import torch
from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_qwen3
from lightning.pytorch import Callback, LightningModule
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger as _TensorBoardLogger
from tbparse import SummaryReader
from torch import Tensor
from torch.optim.adamw import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerFast  # type: ignore
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION, get_scheduler

from primer.callbacks.gradient_accumulation import GradientAccumulationScheduler
from primer.utilities import DictConfig, get_logger

logger = get_logger("model")

TYPE_TO_OPTIMIZER_CLASS = {"adamw": AdamW}


def get_model_config(model_config: dict, tok: PreTrainedTokenizerFast, use_flex_attention: bool = False) -> dict:
    attn_implementation = (
        "flex_attention"
        if use_flex_attention
        else "flash_attention_2"
        if importlib.util.find_spec("flash_attn")
        else "sdpa"
    )

    kwargs = {
        "vocab_size": len(tok),
        "bos_token_id": tok.bos_token_id,  # type: ignore
        "eos_token_id": tok.eos_token_id,  # type: ignore
        "pad_token_id": tok.pad_token_id,  # type: ignore
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "max_position_embeddings": 2048,
        "_attn_implementation": attn_implementation,
    }
    return {**kwargs, **model_config}


def load_hf_from_pl(checkpoint_path: str | Path) -> PreTrainedModel:
    logger.info(f"Reading checkpoint from {checkpoint_path=}")
    checkpoint = torch.load(
        str(checkpoint_path),
        weights_only=False,
        map_location="cpu",
        # mmap=True,
    )
    state_dict = {
        k.removeprefix("model.").removeprefix("_orig_mod."): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    config = checkpoint["hyper_parameters"].get("config")
    if isinstance(config, dict):
        config = Qwen3Config(**config) if config["model_type"] == "qwen3" else LlamaConfig(**config)
    else:
        # If config is not a dict, it is already a config object
        assert isinstance(config, Qwen3Config | LlamaConfig), "Config should be either Qwen3Config or LlamaConfig"

    model = Qwen3ForCausalLM(config) if config.model_type == "qwen3" else LlamaForCausalLM(config)

    model.load_state_dict(state_dict)
    logger.info(f"Model {config=}")
    return model


class RunningStage(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


@dataclass
class OptimCofig(DictConfig):
    # Optimizer config
    optim_name: str
    lr: float
    weight_decay: float = 0.0
    optim_kwargs: dict = field(default_factory=dict)

    # Scheduler config
    scheduler_name: str | None = None
    num_warmup_steps: int | None = None
    scheduler_kwargs: dict = field(default_factory=dict)

    # Gradient accumulation config
    grad_acc_schedule: dict | None = None
    zloss_factor: float | None = None  # lambda for zloss, if used

    def __post_init__(self) -> None:
        assert self.optim_name in TYPE_TO_OPTIMIZER_CLASS
        if self.scheduler_name is not None:
            assert self.scheduler_name in TYPE_TO_SCHEDULER_FUNCTION


class LanguageModel(LightningModule):
    def __init__(
        self, config: dict, optim_config: OptimCofig, use_torch_compile: bool = False, use_liger: bool = False
    ) -> None:
        super().__init__()
        # Asking for config to be dict so that lightning saves it in the checkpoint without problems
        self.config = Qwen3Config(**config) if config["model_type"] == "qwen3" else LlamaConfig(**config)
        self.optim_config = optim_config
        self.use_torch_compile = use_torch_compile
        self.use_liger = use_liger
        self.save_hyperparameters()

    def configure_model(self) -> None:
        self.model = (
            LlamaForCausalLM(self.config) if self.config.model_type == "llama" else Qwen3ForCausalLM(self.config)
        )

        if self.use_liger:
            apply_liger_kernel = (
                apply_liger_kernel_to_llama if self.config.model_type == "llama" else apply_liger_kernel_to_qwen3
            )
            apply_liger_kernel(
                rope=True,
                cross_entropy=False,
                fused_linear_cross_entropy=True,
                rms_norm=True,
                swiglu=True,
                model=self.model,
            )

        if self.use_torch_compile or self.config._attn_implementation == "flex_attention":
            logger.info("Using torch.compile. This is triggered even if you set it to False but use flex_attention.")
            self.model = torch.compile(self.model)

        logger.info(
            f"Model config:\n{self.model.config.to_json_string()}\n"
            f"Attention implementation: {self.model.config._attn_implementation}\n"
            f"Memory footprint: {self.model.get_memory_footprint() / 1e6:.2f}MB\n"
            f"Num parameters: {self.model.num_parameters() / 1e6:.1f}M"
        )

    def configure_callbacks(self) -> list[Callback]:
        # These are optimisation-related callbacks, so I want to initialise them here
        return [GradientAccumulationScheduler(scheduling=self.optim_config.grad_acc_schedule)]

    def on_train_start(self) -> None:
        # Log hyperparameters to TensorBoard: https://lightning.ai/docs/pytorch/latest/extensions/logging.html#logging-hyperparameters
        if isinstance(self.logger, TensorBoardLogger):
            hp_metrics = {f"{stage}/loss": float("Inf") for stage in [RunningStage.TRAIN, RunningStage.VALIDATION]}
            self.logger.log_hyperparams(self.hparams, hp_metrics)  # type: ignore

    def forward(self, input_ids: Tensor, **kwargs) -> Tensor:
        return self.model.forward(input_ids=input_ids, **kwargs).logits  # type: ignore

    def step(self, batch: dict[str, Tensor], stage: RunningStage) -> Tensor | None:
        # input_ids = batch["input_ids"][..., :-1].contiguous()
        # att_mask = batch["att_mask"][..., :-1].contiguous() if "att_mask" in batch else None
        # labels = batch["input_ids"][..., 1:].contiguous()
        # logits = self.forward(input_ids=input_ids, attention_mask=att_mask)
        # logits = logits.float()  # Upcast to float to avoid potential precision issues
        # loss = cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        # =====================================================================================================
        # Above is functionally equivalent to this below, but HF implements it with padding instead of indexing
        # which, strangely enough, is faster and more memory-efficient than indexing + .contiguous().
        # =====================================================================================================
        out = self.model(
            input_ids=batch["input_ids"], labels=batch["input_ids"].clone(), attention_mask=batch.get("att_mask")
        )
        loss = out.loss
        logs = {"loss": loss.detach()}
        if self.optim_config.zloss_factor is not None:
            logits = out.logits
            zloss = logits.logsumexp(dim=-1).pow(2).mean()
            loss += self.optim_config.zloss_factor * zloss
            logs["zloss"] = zloss.detach()
            logs["total_loss"] = loss.detach()

        self.log_dict(
            {f"{stage}/{k}": v for k, v in logs.items()},
            on_step=stage == RunningStage.TRAIN,
            on_epoch=stage == RunningStage.VALIDATION,
            prog_bar=True,
            logger=True,
            batch_size=batch["input_ids"].shape[0],
            sync_dist=True,
        )

        if stage == RunningStage.TRAIN:
            return loss

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        return self.step(batch, RunningStage.TRAIN)  # type: ignore

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        return self.step(batch, RunningStage.VALIDATION)  # type: ignore

    def configure_optimizers(self) -> dict:
        # Get params that require grad
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}

        # Create optim_groups taking care to check whether we want the keller optimiser
        decay_params, nodecay_params = [], []
        for _, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        logger.info(f"{len(decay_params)=}, with {sum(p.numel() for p in decay_params):,} params")
        logger.info(f"{len(nodecay_params)=}, with {sum(p.numel() for p in nodecay_params):,} params")

        optim_groups = [
            {"params": decay_params, "weight_decay": self.optim_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        out = {}

        # Create optimizer
        opt = TYPE_TO_OPTIMIZER_CLASS[self.optim_config.optim_name](
            optim_groups, lr=self.optim_config.lr, **self.optim_config.optim_kwargs
        )
        out["optimizer"] = opt

        # Maybe create scheduler
        if self.optim_config.scheduler_name is not None:
            lr_scheduler = get_scheduler(
                name=self.optim_config.scheduler_name,
                num_warmup_steps=self.optim_config.num_warmup_steps,
                optimizer=opt,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                scheduler_specific_kwargs=self.optim_config.scheduler_kwargs,
            )
            out["lr_scheduler"] = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return out


class TensorBoardLogger(_TensorBoardLogger):
    LOGGER_NAME: str = "tensorboard"

    def __init__(self, save_dir: str | Path, name: str | None = "tb_logs", version: int | str | None = None) -> None:
        # Set this to default_hp_metric=False since we log_hparams manually `on_train_start`
        super().__init__(save_dir=save_dir, name=name, version=version, default_hp_metric=False)

    @property
    def logger_name(self) -> str:
        return self.LOGGER_NAME

    def save_to_parquet(self, path: str | Path) -> None:
        SummaryReader(str(self.log_dir)).scalars.to_parquet(path)


# =====================================================================================================================
# I tried the approach but it is slower and more memory-intensive because it's implementing log-softmax + NLL manually,
# whereas torch.nn.functional.cross_entropy is a fused and highly optimized kernel (implemented in C++) that combines
# log_softmax and nll_loss to avoid intermediate allocations and save memory.
# =====================================================================================================================
# class CrossEntropyWithZLoss(torch.nn.Module):
#     """ZLoss is a modified CrossEntropyLoss.

#     When z_loss=0: they are equivalent. z_loss encourages the logits:
#     - to not drift too far from zero (which can cause unacceptable roundoff errors in bfloat16)
#     - to be normalized log-probabilities
#     Based on t5x and mesh_tensorflow implementations:
#     https://github.com/google-research/t5x/blob/77d2624e65799e3bea15586eb1d3fe7c63477a92/t5x/models.py#L738
#     https://github.com/google-research/t5x/blob/0728d8429041d6c6e75077334e76eb2370c6057b/t5x/losses.py#L50
#     https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
#     and https://github.com/Birch-san/z-loss-pytorch/blob/main/z_loss.py
#     """

#     def __init__(
#         self,
#         ignore_index: int = -100,
#         reduction: Literal["mean", "sum", "none"] = "mean",
#         zloss_factor: float | None = None,
#     ) -> None:
#         super().__init__()
#         self.ignore_index = ignore_index

#         assert reduction in {"mean", "sum", "none"}, f"Invalid reduction: {reduction}"
#         self.reduction = reduction

#         assert zloss_factor is None or (isinstance(zloss_factor, float) and zloss_factor >= 0), (
#             f"Invalid zloss_factor: {zloss_factor}"
#         )
#         self.zloss_factor = zloss_factor

#     def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> Tensor:
#         """To keep the interface similar to the original cross_entropy, we expect flattened logits and labels.

#         logits: (B, V), labels: (B,)
#         """
#         # log_z = logits.logsumexp(dim=-1)  # (B, T, V) -> (B, T)
#         # log_softmax = logits - log_z.unsqueeze(-1)  # (B, T, V) - (B, T, 1) -> (B, T, V)
#         # loss = nll_loss(log_softmax.flatten(end_dim=-2), labels.flatten(), ignore_index=self.ignore_index, reduction='none')  # (B, T, V) -> (B, T)
#         # loss = loss.unflatten(0, labels.shape)  # (B, T)
#         log_z = logits.logsumexp(dim=-1)  # (B,)
#         log_softmax = logits - log_z.unsqueeze(-1)  # (B, V)
#         loss = nll_loss(log_softmax, labels, ignore_index=self.ignore_index, reduction="none")  # (B,)

#         if self.zloss_factor is not None:
#             zloss = log_z.masked_fill(labels == self.ignore_index, 0).pow(2)
#             loss += self.zloss_factor * zloss

#         if self.reduction == "none":
#             return loss

#         loss = loss.sum()
#         if self.reduction == "sum":
#             return loss

#         nonignored_token_count = labels.numel() - (labels == self.ignore_index).sum()
#         loss /= nonignored_token_count
#         return loss


# def zloss_with_logs(logits: Tensor, labels: Tensor, zloss_factor: float) -> tuple[Tensor, dict]:
#     log_z = logits.logsumexp(dim=-1)  # (B,)
#     log_softmax = logits - log_z.unsqueeze(-1)  # (B, V)
#     loss = nll_loss(log_softmax, labels, ignore_index=-100, reduction="mean")  # (B,)
#     logs = {"loss": loss.detach()}

#     zloss = log_z[labels != -100].pow(2).mean()
#     logs["z_loss"] = zloss.detach()

#     loss += zloss_factor * zloss
#     logs["total_loss"] = loss.detach()

#     return loss, logs


# def cross_entropy_with_logs(logits: Tensor, labels: Tensor) -> tuple[Tensor, dict]:
#     """A simple cross entropy loss with zloss factor, returning the loss and logs."""
#     loss = cross_entropy(logits, labels, ignore_index=-100, reduction="mean")
#     return loss, {"loss": loss.detach()}
