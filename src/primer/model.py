import importlib.util
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger as _TensorBoardLogger
from tbparse import SummaryReader
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim.adamw import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerFast  # type: ignore
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION, get_scheduler

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
        "vocab_size": tok.vocab_size,
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
    config = LlamaConfig(**config) if isinstance(config, dict) else config  # TODO: check this
    model = LlamaForCausalLM(config)
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
    keller_kwargs: dict = field(default_factory=dict)

    # Scheduler config
    scheduler_name: str | None = None
    num_warmup_steps: int | None = None
    scheduler_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.optim_name in TYPE_TO_OPTIMIZER_CLASS
        if self.scheduler_name is not None:
            assert self.scheduler_name in TYPE_TO_SCHEDULER_FUNCTION


class LanguageModel(LightningModule):
    def __init__(self, config: dict, optim_config: OptimCofig, use_torch_compile: bool = False) -> None:
        super().__init__()
        # Asking for config to be dict so that lightning saves it in the checkpoint without problems
        self.config = LlamaConfig(**config)
        self.use_torch_compile = use_torch_compile
        self.optim_config = optim_config
        self.save_hyperparameters()

    def configure_model(self) -> None:
        self.model = LlamaForCausalLM(self.config)

        if self.use_torch_compile or self.config._attn_implementation == "flex_attention":
            logger.info("Using torch.compile. This is triggered even if you set it to False but use flex_attention.")
            self.model = torch.compile(self.model)

        logger.info(
            f"Model config:\n{self.model.config.to_json_string()}\n"
            f"Attention implementation: {self.model.config._attn_implementation}\n"
            f"Memory footprint: {self.model.get_memory_footprint() / 1e6:.2f}MB\n"
            f"Num parameters: {self.model.num_parameters() / 1e6:.1f}M"
        )

    def on_train_start(self) -> None:
        # Log hyperparameters to TensorBoard: https://lightning.ai/docs/pytorch/latest/extensions/logging.html#logging-hyperparameters
        if isinstance(self.logger, TensorBoardLogger):
            loss_metrics = {f"{stage}/loss": 0.0 for stage in (RunningStage.TRAIN, RunningStage.VALIDATION)}
            self.logger.log_hyperparams(self.hparams, loss_metrics)  # type: ignore

    def forward(self, input_ids: Tensor, **kwargs) -> Tensor:
        return self.model.forward(input_ids=input_ids, **kwargs).logits  # type: ignore

    def step(self, batch: dict[str, Tensor], stage: RunningStage) -> Tensor | None:
        input_ids = batch["input_ids"][:, :-1]
        att_mask = batch["att_mask"][:, :-1] if "att_mask" in batch else None

        labels = batch["input_ids"][:, 1:].clone()

        logits = self.forward(input_ids=input_ids, attention_mask=att_mask)
        loss = cross_entropy(logits.permute(0, 2, 1), labels)

        self.log(
            f"{stage}/loss",
            loss.detach(),
            on_step=stage == RunningStage.TRAIN,
            on_epoch=stage == RunningStage.VALIDATION,
            prog_bar=True,
            logger=True,
            batch_size=labels.shape[0],
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
