# import json
import copy
import logging
import os
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import colorlog
import polars as pl
import srsly
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rich import print


def set_hf_paths() -> None:
    # Set this here in order to have effect
    # See: https://github.com/huggingface/transformers/issues/25305#issuecomment-1852931139
    # os.environ["TRANSFORMERS_CACHE"] = "./.huggingface_cache"
    CACHE_PATH = "./.huggingface_cache"
    os.environ["HF_HOME"] = CACHE_PATH
    os.environ["HF_DATASETS_CACHE"] = CACHE_PATH
    os.environ["TORCH_HOME"] = CACHE_PATH
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Print the environment variables
    print("HF_HOME:", os.environ["HF_HOME"])
    print("HF_DATASETS_CACHE:", os.environ["HF_DATASETS_CACHE"])
    print("TORCH_HOME:", os.environ["TORCH_HOME"])
    print("TOKENIZERS_PARALLELISM:", os.environ["TOKENIZERS_PARALLELISM"])
    print("HF_HUB_ENABLE_HF_TRANSFER:", os.environ["HF_HUB_ENABLE_HF_TRANSFER"])


@contextmanager
def track_time(desc: str, task_name: str = "task") -> Generator[None, Any, None]:
    print(desc)
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        print(f"⏱️ task=`{task_name}` completed in {elapsed_time:.2f} seconds")


@dataclass
class DictConfig:
    """Dataclass which is subscriptable like a dict"""

    def to_dict(self) -> dict[str, Any]:
        out = copy.deepcopy(self.__dict__)
        return out

    def __getitem__(self, k: str) -> Any:
        return self.__dict__[k]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)


def get_logger(name: str, level: Literal["error", "warning", "info", "debug"] = "info") -> logging.Logger:
    # Convert the level string to the corresponding logging level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure the logger and configure colorlog
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"  # noqa: E501
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def flatten(x: list[list]) -> list:
    return [i for j in x for i in j]


def remove_file(path: str | Path) -> None:
    path = Path(path)
    path.unlink(missing_ok=True)


def jsonl2parquet(filepath: str | Path, out_dir: str | Path) -> None:
    filepath = Path(filepath)
    assert filepath.name.endswith(".jsonl"), "Not a jsonl file"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fl = srsly.read_jsonl(filepath)
    df = pl.DataFrame({k: flatten(v) for k, v in ld_to_dl(line).items()} for line in fl)  # type: ignore
    df = df.explode(df.columns)

    df.write_parquet(out_dir / f"{filepath.name.removesuffix('.jsonl')}.parquet")


def ld_to_dl(ld: list[dict]) -> dict[str, list]:
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def conf_to_dict(x: DictConfig | None) -> dict:
    if x is not None:
        return OmegaConf.to_container(x)  # type: ignore
    return {}


def instantiate_from_conf(list_cfg: list[DictConfig]) -> list:
    return [list(instantiate(cfg).values()) if cfg is not None else None for cfg in list_cfg]
