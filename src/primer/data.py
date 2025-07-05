from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.compute as pc
import torch
from datasets import Dataset, load_from_disk
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset as TorchDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from primer.utilities import DictConfig, get_logger

logger = get_logger("data")


class OffsetLocator:
    def __init__(self, offsets: np.ndarray, block_size: int = 2048) -> None:
        assert offsets.ndim == 1, f"Expected offsets to be a 1D array, but got {offsets.ndim}D array."
        self.offsets = offsets
        self.block_size = block_size

        # Trim or pad offsets to fit blocks
        self.total_len = len(offsets)
        pad_len = (block_size - self.total_len % block_size) % block_size
        if pad_len > 0:
            # Pad with a large number to avoid false positives
            offsets = np.concatenate([offsets, np.full(pad_len, offsets[-1] + 1)])

        self.offsets_2d = offsets.reshape(-1, block_size)
        self.block_starts = self.offsets_2d[:, 0]
        assert np.all(self.block_starts[:-1] <= self.block_starts[1:]), "offsets must be sorted"

    def locate(self, pos: int) -> int:
        """Return the index i such that offsets[i] <= pos < offsets[i+1]."""
        # Level 1: Find block
        block_idx = np.searchsorted(self.block_starts, pos, side="right") - 1
        block_idx = np.clip(block_idx, 0, self.offsets_2d.shape[0] - 1)

        # Level 2: Search within block
        row = self.offsets_2d[block_idx]
        within_idx = np.searchsorted(row, pos, side="right") - 1
        final_idx = block_idx * self.block_size + within_idx

        # Clip to avoid going past original array
        return min(final_idx, self.total_len - 1)

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens in the offsets."""
        return int(self.offsets[-1])

    def get(self, idx: int) -> int:
        """Get the offset at the given index."""
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError(f"Index {idx} is out of bounds for offsets with length {len(self.offsets)}.")
        return int(self.offsets[idx])

    def __len__(self) -> int:
        """Return the number of offsets."""
        return len(self.offsets)


class PackedTokenDataset(TorchDataset):
    """A map-style dataset that packs tokenized documents into fixed-length sequences.

    The packing works by concatenating documents (potentially shuffling them first). Once
    the sequences are formed, then they are potentially shuffled.

    The dataset is expected to be a Hugging Face dataset, and the `input_ids` column
    """

    # NOTE: hard-coding this since limit is 4,294,967,295. If your data is super big, change this.
    _idx_dtype: np.typing.DTypeLike = np.uint32

    # Using uint64 to avoid overflow issues with large datasets (as tokens can easily be in the trillions)
    _offsets_dtype: np.typing.DTypeLike = np.uint64

    def __init__(
        self,
        data_path: str | Path,
        seq_len: int,
        eod_token_id: int = 0,
        shuffle_seed: int | None = None,
        intra_doc_causal_mask: bool = False,
    ) -> None:
        """
        Args:
            data_path (str | Path): Path to a Hugging Face datasets.Dataset.
            seq_len (int): Max context length used by the model. Internally, this value is incremented by 1
                to account for the end-of-document (EOD) token, resulting in sequences of length `seq_len + 1`.
                This adjustment ensures proper handling of the EOD token during training.
            eod_token_id (int): End-of-document token (default: 0).
            shuffle_seed (int | None): Seed for shuffling documents and sequences.
                If None, no shuffling is performed.
            intra_doc_causal_mask (bool): Whether to apply a causal mask within individual documents.
                If True, tokens within the same document will only attend to previous tokens in the sequence.
        """
        self.data_path = Path(data_path)
        self.seq_len = seq_len + 1  # Add one because during training you need shift by one to compute loss
        self.eod_token_id = eod_token_id
        self.shuffle_seed = shuffle_seed
        self.intra_doc_causal_mask = intra_doc_causal_mask
        self.setup()

    def setup(self) -> None:
        # Read datasets
        assert self.data_path.exists(), f"Data path {self.data_path} does not exist."
        self.dataset: Dataset = load_from_disk(str(self.data_path))  # type: ignore
        assert "input_ids" in self.dataset.column_names

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
        offsets = self._load_memmap(offsets_path, is_offsets=True)
        self.offsets = OffsetLocator(offsets, block_size=2048)

        # Calculate total number of sequences
        self.total_tokens = self.offsets.total_tokens
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

        if self.intra_doc_causal_mask:
            logger.info("Using intra-document causal attention mask")

    def _format_metadata_path(self, filename: str) -> Path:
        suffix = f"seed{self.shuffle_seed}" if self.shuffle_seed is not None else "noshuffle"
        suffix += f"_eod{self.eod_token_id}_seq{self.seq_len}.npy"
        return self.data_path / f"{filename}_{suffix}"

    def _save_arange_memmap(self, size: int, path: str | Path) -> None:
        # Moved to doing this in RAM instead of directly writing to memmap,
        arr = np.arange(size, dtype=self._idx_dtype)

        if self.shuffle_seed is not None:
            rng = np.random.default_rng(self.shuffle_seed)
            rng.shuffle(arr)  # Shuffle directly in the memory-mapped file

        memmap = np.memmap(path, dtype=self._idx_dtype, mode="w+", shape=(size,))
        memmap[:] = arr
        memmap.flush()
        assert len(memmap) == size, f"Expected {size} elements, but got {len(memmap)}."

    def _save_offsets_memmap(self, path: str | Path) -> None:
        """Compute document boundary, i.e., offsets.

        If we are shuffling documents, we first shuffle them and then "concatenate" (in the cumsum) and
        compute the offsets.
        """
        doc_lens = pc.list_value_length(self.dataset.data.table["input_ids"]).to_numpy()  # type: ignore

        assert doc_lens.ndim == 1, f"Expected 1D array for document lengths, but got {doc_lens.ndim}D."
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

    # def get_sequence(self, start_pos: int, end_pos: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    #     """Retrieve a sequence with minimal overhead and fast memory operations."""
    #     current_tokens = torch.empty(end_pos - start_pos, dtype=torch.long)
    #     att_mask_doc_ids = torch.empty(self.seq_len, dtype=torch.long) if self.intra_doc_causal_mask else None

    #     # Vectorized operations to retrieve tokens
    #     positions = torch.arange(start_pos, end_pos, dtype=torch.long)
    #     shuffled_doc_indices = torch.tensor([self.offsets.locate(pos.item()) for pos in positions], dtype=torch.long)
    #     doc_indices = torch.tensor([int(self.doc_idx[idx]) for idx in shuffled_doc_indices], dtype=torch.long)

    #     for i, doc_idx in enumerate(doc_indices.unique_consecutive()):
    #         input_ids = torch.tensor(self.dataset[doc_idx.item()]["input_ids"], dtype=torch.long)
    #         mask = doc_indices == doc_idx
    #         relative_positions = positions[mask] - self.offsets.get(shuffled_doc_indices[mask][0].item())
    #         current_tokens[mask] = input_ids[relative_positions]

    #         if att_mask_doc_ids is not None:
    #             att_mask_doc_ids[mask] = doc_idx

    #     # Add EOD token if necessary
    #     if current_tokens.size(0) < self.seq_len:
    #         current_tokens[current_tokens.size(0)] = self.eod_token_id

    #     return current_tokens, att_mask_doc_ids

    def get_sequence(self, start_pos: int, end_pos: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Retrieve a sequence with minimal overhead and fast memory operations."""
        current_tokens = torch.empty(end_pos - start_pos, dtype=torch.long)
        att_mask_doc_ids = torch.empty(self.seq_len, dtype=torch.long) if self.intra_doc_causal_mask else None
        pos = start_pos
        i = 0
        while pos < end_pos:
            # Find the relevant document for the current position in the shuffled order using binary search
            # Again, since we are shuffling, the document index is with respect to the shuffled documents
            shuffled_doc_idx = self.offsets.locate(pos)

            # Since self.dataset is NOT shuffled, we need the document position in the original non-shuffled order
            doc_idx = int(self.doc_idx[shuffled_doc_idx])

            # Now, get the document's input_ids
            input_ids = self.dataset[doc_idx]["input_ids"]

            # Get relative position of pos within the document
            # Remember pos is the absolute position within the entire "stream" of tokens
            doc_start = int(pos - self.offsets.get(shuffled_doc_idx))

            # Get the number of tokens to copy from the document
            tokens_to_copy = min(len(input_ids) - doc_start, end_pos - pos)

            # Use PyTorch's in-place assignment without slicing
            current_tokens[i : i + tokens_to_copy] = torch.as_tensor(
                input_ids[doc_start : doc_start + tokens_to_copy], dtype=torch.long
            )

            if att_mask_doc_ids is not None:
                att_mask_doc_ids[i : i + tokens_to_copy] = doc_idx  # Each token gets the document index

            # Update counters
            i += tokens_to_copy
            pos += tokens_to_copy

            # Add EOD token if the document ends, and more tokens are needed
            if doc_start + tokens_to_copy == len(input_ids) and pos < end_pos:
                current_tokens[i] = self.eod_token_id
                i += 1
                pos += 1

        return current_tokens, att_mask_doc_ids

    def build_intra_doc_causal_mask(self, doc_ids: torch.Tensor) -> torch.Tensor:
        """Intra-document causal attention mask.

        Builds an attention mask where token i can only attend to token j if:
        - j <= i (causal)
        - doc_ids[i] == doc_ids[j] (intra-doc)
        """
        seq_len = doc_ids.size(0)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        doc_mask = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
        return (causal_mask & doc_mask).long()

    def __len__(self) -> int:
        """Return the total number of sequences."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve the sequence at the given index."""
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(f"Sequence index {idx} is out of bounds for dataset with {self.num_sequences} sequences.")

        # Possibly shuffle at the sequence level
        # Converting to int as it might overflow when index is np.uint32
        index = int(self.seq_idx[idx])
        start_pos = index * self.seq_len
        end_pos = start_pos + self.seq_len

        tokens, att_mask = self.get_sequence(start_pos, end_pos)
        out = {"input_ids": tokens}
        if att_mask is not None:
            out["att_mask"] = att_mask  # self.build_intra_doc_causal_mask(att_mask)
        return out


@dataclass
class DataloaderConfig(DictConfig):
    batch_size: int | None = None
    eval_batch_size: int | None = None
    shuffle_seed: int | None = None
    intra_doc_causal_mask: bool = False

    # kwargs
    num_workers: int | None = cpu_count()
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = False
    multiprocessing_context: str | None = None
    prefetch_factor: int | None = None

    def get_dataloader_kwargs(self) -> dict:
        # Remove batch_size, eval_batch_size, and shuffle_seed from the dataloader configuration
        kwargs = {
            k: v
            for k, v in self.to_dict().items()
            if k not in ["batch_size", "eval_batch_size", "shuffle_seed", "intra_doc_causal_mask"]
        }

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
        seq_len: int,
        eod_token_id: int,
        dataloader_config: DataloaderConfig,
    ) -> None:
        super().__init__()
        self.train_data_path = Path(train_data_path) if train_data_path else train_data_path
        self.val_data_path = Path(val_data_path) if val_data_path else val_data_path
        self.seq_len = seq_len
        self.eod_token_id = eod_token_id
        self.dataloader_config = dataloader_config
        self.save_hyperparameters()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None) -> None:
        if self.train_data_path:
            self.train_ds = PackedTokenDataset(
                data_path=str(self.train_data_path),
                seq_len=self.seq_len,
                eod_token_id=self.eod_token_id,
                shuffle_seed=self.dataloader_config.shuffle_seed,
                intra_doc_causal_mask=self.dataloader_config.intra_doc_causal_mask,
            )
            logger.info(f"Train dataset loaded: {len(self.train_ds)=}")
            logger.info(f"{self.train_ds=}")

        if self.val_data_path:
            self.val_ds = PackedTokenDataset(
                data_path=str(self.val_data_path),
                seq_len=self.seq_len,
                eod_token_id=self.eod_token_id,
                intra_doc_causal_mask=self.dataloader_config.intra_doc_causal_mask,
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
