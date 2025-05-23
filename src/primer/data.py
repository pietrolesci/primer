from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from datasets import Dataset, load_from_disk
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset as TorchDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from primer.utilities import DictConfig, get_logger

logger = get_logger("data")


class PackedTokenDataset(TorchDataset):
    """A map-style dataset that packs tokenized documents into fixed-length sequences.

    The packing works by concatenating documents (potentially shuffling them first). Once
    the sequences are formed, then they are potentially shuffled.

    The dataset is expected to be a Hugging Face dataset, and the `input_ids` column
    """

    # NOTE: hard-coding this since limit is 4,294,967,295. If your data is super big, change this.
    _idx_dtype: np.typing.DTypeLike = np.uint32

    # Directly save the offsets to a memory-mapped file
    # Using uint64 to avoid overflow issues with large datasets (as tokens can easily be in the trillions)
    _offsets_dtype: np.typing.DTypeLike = np.uint64

    def __init__(
        self,
        data_path: str | Path,
        seq_len: int,
        eod_token_id: int = 0,
        shuffle_seed: int | None = None,
        num_tokens_col: str | None = None,
    ) -> None:
        """
        Args:
            data_path (str | Path): Path to a Hugging Face datasets.Dataset.
            seq_len (int): Max context length used by the model. This value will be incremented by 1 internally
                to account for the end-of-document (EOD) token, resulting in sequences of length `seq_len + 1`.
            eod_token_id (int): End-of-document token (default: 0).
            shuffle_seed (int | None): Seed for shuffling documents and sequences.
                If None, no shuffling is performed.
            num_tokens_col (str | None): Column name for the number of tokens in the dataset.
                If None, it will be computed.
        """
        self.seq_len = seq_len + 1  # Add one because during training you need shift by one to compute loss
        self.eod_token_id = eod_token_id
        self.num_tokens_col = num_tokens_col
        self.shuffle_seed = shuffle_seed
        self.data_path = Path(data_path)

        self.setup()

    def setup(self) -> None:
        # Read datasets
        assert self.data_path.exists(), f"Data path {self.data_path} does not exist."
        self.dataset: Dataset = load_from_disk(str(self.data_path))  # type: ignore
        assert "input_ids" in self.dataset.column_names, "Dataset must contain 'input_ids' column."

        # Creating doc_idx to control document shuffling
        doc_idx_path = self._format_metadata_path("docs")
        if not doc_idx_path.exists():
            logger.info(f"Creating document shuffling indices and saving to `{doc_idx_path}`")
            if self.shuffle_seed is not None:
                logger.info(f"{self.shuffle_seed=} therefore shuffling the documents")
            self._save_arange_memmap(len(self.dataset), doc_idx_path)
        logger.info(f"Loaded document shuffling indices from `{doc_idx_path}`")
        self.doc_idx = self._load_memmap(doc_idx_path)

        # Creating document offsets for sequence packing
        offsets_path = self._format_metadata_path("offsets")
        if not offsets_path.exists():
            logger.info(f"Computing offsets and saving to `{offsets_path}`")
            self._save_offsets_memmap(offsets_path)
        logger.info(f"Loaded offsets from `{offsets_path}`")
        self.offsets = self._load_memmap(offsets_path, is_offsets=True)

        # Calculate total number of sequences
        self.total_tokens = int(self.offsets[-1])
        self.num_sequences = self.total_tokens // self.seq_len  # Drops remainder of tokens that do not fill seq_len

        # Computing seq_idx to control sequence shuffling
        seq_idx_path = self._format_metadata_path("seqs")
        if not seq_idx_path.exists():
            logger.info(f"Creating sequence shuffling indices and saving to `{seq_idx_path}`")
            if self.shuffle_seed is not None:
                logger.info(f"{self.shuffle_seed=} therefore shuffling the sequences")
            self._save_arange_memmap(self.num_sequences, seq_idx_path)
        logger.info(f"Loaded sequence shuffling indices from `{seq_idx_path}`")
        self.seq_idx = self._load_memmap(seq_idx_path)

    def _format_metadata_path(self, filename: str) -> Path:
        suffix = f"seed{self.shuffle_seed}" if self.shuffle_seed is not None else "noshuffle"
        suffix += f"_eod{self.eod_token_id}_seq{self.seq_len}.npy"
        return self.data_path / f"{filename}_{suffix}"

    def _save_arange_memmap(self, size: int, path: str | Path) -> None:
        memmap = np.memmap(path, dtype=self._idx_dtype, mode="w+", shape=(size,))
        memmap[:] = np.arange(size, dtype=self._idx_dtype)

        if self.shuffle_seed is not None:
            rng = np.random.default_rng(self.shuffle_seed)
            rng.shuffle(memmap)  # Shuffle directly in the memory-mapped file

        memmap.flush()
        assert len(memmap) == size, f"Expected {size} elements, but got {len(memmap)}."

    def _save_offsets_memmap(self, path: str | Path) -> None:
        """Compute document boundary, i.e., offsets.

        If we are shuffling documents, we first shuffle them and then "concatenate" (in the cumsum) and
        compute the offsets. NOTE: I am using polars to extract info from datasets as it is way faster.
        """
        if self.num_tokens_col and self.num_tokens_col in self.dataset.column_names:
            logger.info(f"Using precomputed number of tokens from column {self.num_tokens_col}.")
            doc_lens = pl.from_arrow(self.dataset.data.table[self.num_tokens_col]).to_numpy()  # type: ignore
        else:
            if self.num_tokens_col:
                logger.warning(f"Column {self.num_tokens_col=} passed but not found. Recomputing token counts.")
            doc_lens = pl.from_arrow(self.dataset.data.table["input_ids"]).list.len().to_numpy()  # type: ignore

        doc_lens = doc_lens + 1  # Add 1 for EOD token
        doc_lens = doc_lens[self.doc_idx]  # Reorder document lengths based on shuffled document indices

        # Save directly to a memory-mapped file
        memmap = np.memmap(path, dtype=self._offsets_dtype, mode="w+", shape=(len(doc_lens) + 1,))
        memmap[0] = 0  # Initialize the first offset to 0
        np.cumsum(doc_lens, out=memmap[1:])
        memmap.flush()
        assert len(memmap) == len(doc_lens) + 1, f"Expected {len(doc_lens) + 1} elements, but got {len(memmap)}."

    def _load_memmap(self, path: str | Path, is_offsets: bool = False) -> np.memmap:
        return np.memmap(path, dtype=self._offsets_dtype if is_offsets else self._idx_dtype, mode="r")

    def get_sequence(self, start_pos: int, end_pos: int) -> torch.Tensor:
        """Retrieve a sequence with minimal overhead and fast memory operations."""
        current_tokens = torch.empty(end_pos - start_pos, dtype=torch.long)
        pos = start_pos
        i = 0

        while pos < end_pos:
            # Find the relevant document for the current position in the shuffled order using binary search
            # Again, since we are shuffling, the document index is with respect to the shuffled documents
            shuffled_doc_idx = int(np.searchsorted(self.offsets, pos, side="right") - 1)

            # Since self.dataset is NOT shuffled, we need the document position in the original non-shuffled order
            doc_idx = int(self.doc_idx[shuffled_doc_idx])

            # Now, get the document's input_ids
            input_ids = self.dataset[doc_idx]["input_ids"]

            # Get relative position of pos within the document
            # Remember pos is the absolute position within the entire "stream" of tokens
            doc_start = int(pos - self.offsets[shuffled_doc_idx])

            # Get the number of tokens to copy from the document
            tokens_to_copy = min(len(input_ids) - doc_start, end_pos - pos)

            # Use PyTorch's in-place assignment without slicing
            current_tokens[i : i + tokens_to_copy] = torch.as_tensor(
                input_ids[doc_start : doc_start + tokens_to_copy], dtype=torch.long
            )

            # Update counters
            i += tokens_to_copy
            pos += tokens_to_copy

            # Add EOD token if the document ends, and more tokens are needed
            if doc_start + tokens_to_copy == len(input_ids) and pos < end_pos:
                current_tokens[i] = self.eod_token_id
                i += 1
                pos += 1

        return current_tokens

    def __len__(self) -> int:
        """Return the total number of sequences."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieve the sequence at the given index."""
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(f"Sequence index {idx} is out of bounds for dataset with {self.num_sequences} sequences.")

        # Possibly shuffle at the sequence level
        # Converting to int as it might overflow when index is np.uint32
        index = int(self.seq_idx[idx])
        start_pos = index * self.seq_len
        end_pos = start_pos + self.seq_len
        return self.get_sequence(start_pos, end_pos)


@dataclass
class DataloaderConfig(DictConfig):
    batch_size: int | None = None
    eval_batch_size: int | None = None
    shuffle_seed: int | None = None
    num_workers: int | None = cpu_count()
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = False
    multiprocessing_context: str | None = None
    prefetch_factor: int | None = None

    def get_dataloader_kwargs(self) -> dict:
        # Remove batch_size, eval_batch_size, and shuffle_seed from the dataloader configuration
        kwargs = {k: v for k, v in self.to_dict().items() if k not in ["batch_size", "eval_batch_size", "shuffle_seed"]}

        # NOTE: Shuffling is handled by the PackedTokenDataset to ensure that the sequence-level and
        # document-level shuffling are consistent and reproducible across different runs. This design
        # decision avoids potential conflicts or redundant operations that could arise if the dataloader
        # also attempted to shuffle the data. Therefore, the dataloader will not shuffle.
        kwargs["shuffle"] = False

        return kwargs


class DataModule(LightningDataModule):
    """Data module for tokenized and packed documents into fixed-length sequences.

    Shuffling is handled by the underlying PackedTokenDataset class.
    """

    train_ds: PackedTokenDataset
    val_ds: PackedTokenDataset

    def __init__(
        self,
        train_data_path: str | Path | None,
        val_data_path: str | Path | None,
        max_position_embeddings: int,
        eod_token_id: int,
        dataloader_config: DataloaderConfig,
        num_tokens_col: str | None = None,
    ) -> None:
        super().__init__()
        self.train_data_path = Path(train_data_path) if train_data_path else train_data_path
        self.val_data_path = Path(val_data_path) if val_data_path else val_data_path
        self.max_position_embeddings = max_position_embeddings
        self.eod_token_id = eod_token_id
        self.dataloader_config = dataloader_config
        self.num_tokens_col = num_tokens_col
        self.save_hyperparameters()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        if self.train_data_path:
            self.train_ds = PackedTokenDataset(
                data_path=str(self.train_data_path),
                seq_len=self.max_position_embeddings,
                eod_token_id=self.eod_token_id,
                shuffle_seed=self.dataloader_config.shuffle_seed,
                num_tokens_col=self.num_tokens_col,
            )
            logger.info(f"Train dataset loaded: {len(self.train_ds)=}")
            logger.info(f"{self.train_ds=}")

        if self.val_data_path:
            self.val_ds = PackedTokenDataset(
                data_path=str(self.val_data_path),
                seq_len=self.max_position_embeddings,
                eod_token_id=self.eod_token_id,
                num_tokens_col=self.num_tokens_col,
            )
            logger.info(f"Validation dataset loaded: {len(self.val_ds)=}")
            logger.info(f"{self.val_ds=}")

    def train_dataloader(self) -> StatefulDataLoader:
        return StatefulDataLoader(
            self.train_ds,
            batch_size=self.dataloader_config["batch_size"],
            **self.dataloader_config.get_dataloader_kwargs(),
        )

    def val_dataloader(self) -> StatefulDataLoader:
        return StatefulDataLoader(
            self.val_ds,
            batch_size=self.dataloader_config["eval_batch_size"],
            **self.dataloader_config.get_dataloader_kwargs(),
        )
