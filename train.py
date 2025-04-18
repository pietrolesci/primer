import logging
from pathlib import Path

import hydra
import torch
from lightning import Trainer, seed_everything
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # type: ignore

from primer.data import DataloaderConfig, DataModule
from primer.model import get_model
from primer.trainer import LanguageModel, OptimCofig, TensorBoardLogger
from primer.utilities import add_rich_handler, conf_to_dict, instantiate_from_conf, track_time

SEP_LINE = f"{'=' * 80}"


# Configure the logger and configure colorlog
logger = logging.getLogger("hydra")
logger = add_rich_handler(logger)


@hydra.main(version_base=None, config_path="conf", config_name="train_conf")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, "./hparams.yaml")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}\n{SEP_LINE}")

    # Load tokenizer
    logger.info(f"Loading tokenizer {cfg.tok_path}{'/' + cfg.tok_subfolder if cfg.tok_subfolder else ''}")
    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(cfg.tok_path, subfolder=cfg.tok_subfolder)  # type: ignore
    assert isinstance(tok.eos_token_id, int), "Tokenizer must have an eos_token_id of type int"

    # Load model
    model, config = get_model(cfg.model, tok)
    logger.info(f"Model config:\n{model.config.to_json_string()}")
    logger.info(f"Attention implementation: {model.config._attn_implementation}")
    logger.info(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")
    logger.info(f"Num parameters: {model.num_parameters() / 1e6:.1f}M")

    # Load datamodule
    dataloader_config = DataloaderConfig(**conf_to_dict(cfg.data))
    datamodule = DataModule(
        train_data_path=cfg.train_data_path,
        val_data_path=cfg.val_data_path,
        max_position_embeddings=model.config.max_position_embeddings,
        eod_token_id=tok.eos_token_id,
        dataloader_config=dataloader_config,
    )

    # Maybe compile
    if cfg.torch_compile:
        model = torch.compile(model)

    # Load module
    optim_config = OptimCofig(**conf_to_dict(cfg.optim))  # type: ignore
    module = LanguageModel(model, config, optim_config)  # type: ignore

    # Check if we are in a SLURM environment
    env = SLURMEnvironment()
    plugins = []
    if env.detect():
        logger.info("SLURM detected, adding plugin")
        plugins.append(env)

    # Load trainer
    loggers, callbacks = instantiate_from_conf([cfg.get(i) for i in ("loggers", "callbacks")])
    trainer = Trainer(**conf_to_dict(cfg.trainer), logger=loggers, callbacks=callbacks, plugins=plugins)

    # Train
    ckpt_path = cfg.resume_from_checkpoint if Path(cfg.resume_from_checkpoint).exists() else None
    with track_time("Training"):
        seed_everything(cfg.seed)
        torch.set_float32_matmul_precision("high")
        trainer.fit(model=module, datamodule=datamodule, ckpt_path=ckpt_path)

    for log in trainer.loggers:
        if isinstance(log, TensorBoardLogger):
            log.save_to_parquet("tb_logs.parquet")


if __name__ == "__main__":
    main()
