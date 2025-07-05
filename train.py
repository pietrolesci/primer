from pathlib import Path

import hydra
import torch
from lightning import Trainer, seed_everything
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # type: ignore

from primer.data import DataloaderConfig, DataModule
from primer.model import LanguageModel, OptimCofig, TensorBoardLogger, get_model_config
from primer.utilities import conf_to_dict, get_logger, instantiate_from_conf, track_time

SEP_LINE = f"{'=' * 80}"


# Configure the logger and configure colorlog
logger = get_logger("hydra")


@hydra.main(version_base=None, config_path="conf", config_name="train_conf")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, "./hparams.yaml")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}\n{SEP_LINE}")

    # Load tokenizer
    logger.info(f"Loading tokenizer {cfg.tok_path}{'/' + cfg.tok_subfolder if cfg.tok_subfolder else ''}")
    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(cfg.tok_path, subfolder=cfg.tok_subfolder)  # type: ignore
    assert isinstance(tok.eos_token_id, int), "Tokenizer must have an eos_token_id of type int"

    # Load configs
    dataloader_config = DataloaderConfig(**conf_to_dict(cfg.data))
    model_config = get_model_config(cfg.model, tok, use_flex_attention=dataloader_config.intra_doc_causal_mask)

    # Load datamodule
    datamodule = DataModule(
        train_data_path=cfg.train_data_path,
        val_data_path=cfg.val_data_path,
        seq_len=model_config["max_position_embeddings"],
        eod_token_id=tok.eos_token_id,
        dataloader_config=dataloader_config,
    )

    # Check if we are in a SLURM environment
    env = SLURMEnvironment()
    plugins = []
    if env.detect():
        logger.info("SLURM detected, adding plugin")
        plugins.append(env)

    # Load trainer
    seed_everything(cfg.seed)
    loggers, callbacks = instantiate_from_conf([cfg.get(i) for i in ("loggers", "callbacks")])
    trainer = Trainer(**conf_to_dict(cfg.trainer), logger=loggers, callbacks=callbacks, plugins=plugins)

    # Instantiate the model on device directly
    optim_config = OptimCofig(**conf_to_dict(cfg.optim))  # type: ignore
    with trainer.init_module():
        module = LanguageModel(model_config, optim_config, use_torch_compile=cfg.torch_compile)
        module.configure_model()

    # Train
    ckpt_path = Path(cfg.resume_from_checkpoint) if cfg.resume_from_checkpoint else None
    if ckpt_path and not ckpt_path.exists():
        logger.info(
            f"Checkpoint path {ckpt_path} does not exist (yet). If you are running for the first time "
            "this is fine. Next time you run, this will be the path to the last checkpoint."
        )
        ckpt_path = None

    with track_time("Training"):
        torch.set_float32_matmul_precision("high")
        trainer.fit(model=module, datamodule=datamodule, ckpt_path=ckpt_path)

    with track_time("Validating"):
        # use the last checkpoint for validation
        trainer.validate(model=module, dataloaders=datamodule.val_dataloader(), ckpt_path="last")

    for log in trainer.loggers:
        if isinstance(log, TensorBoardLogger):
            log.save_to_parquet("tb_logs.parquet")


if __name__ == "__main__":
    main()
