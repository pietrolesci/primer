import os

# Set this here in order to have effect
# See: https://github.com/huggingface/transformers/issues/25305#issuecomment-1852931139
os.environ["HF_HOME"] = "./.huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "./.huggingface_cache"
os.environ["TORCH_HOME"] = "./.huggingface_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil
from pathlib import Path

import typer
from datasets import DatasetDict, load_dataset
from datatrove.data import DocumentsPipeline
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers.huggingface import HuggingFaceDatasetWriter
from datatrove.utils.batching import batched
from huggingface_hub import CommitOperationCopy, CommitOperationDelete, HfApi, create_commit
from transformers import AutoTokenizer

app = typer.Typer()

num_train_docs = 20_000_000
num_val_docs = 200_000
REPO_ID = "pietrolesci/finewebedu-20B"
SOURCE_REPO_ID = "hf://datasets/HuggingFaceFW/fineweb-edu/sample/100BT"


def tokenize(
    tok_path: str = "pietrolesci/tokenisers", subfolder: str | None = "bpe32000minipile", batch_size: int = 1000
) -> None:
    class DocumentTokenizer(PipelineStep):
        def __init__(
            self, pretrained_model_name_or_path: str, subfolder: str | None = None, batch_size: int = 1000
        ) -> None:
            super().__init__()
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
            self.batch_size = batch_size

        def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:  # type: ignore
            for batch in batched(data, self.batch_size):
                with self.track_time(unit="batch"):
                    docs = [doc.text for doc in batch]
                    encoded_docs: list[list[int]] = self.tokenizer(docs)["input_ids"]  # type: ignore
                    for doc, encoded in zip(batch, encoded_docs, strict=True):
                        doc.metadata["input_ids"] = encoded
                        # doc.metadata["num_tokens"] = len(encoded)  # for the future: this would have been convenient
                        yield doc

    source_path = f"hf://datasets/{REPO_ID}/data"
    tok_name = subfolder if subfolder is not None else Path(tok_path).name
    print(
        f"Tokenizing {source_path} with {tok_path=}{'/' + subfolder if subfolder else ''} "
        f"with {batch_size=} and saving into {REPO_ID}/{tok_name}"
    )

    pipe_tokenize = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                source_path, file_progress=True, doc_progress=True, shuffle_files=False, text_key="text", id_key="id"
            ),
            DocumentTokenizer(pretrained_model_name_or_path=tok_path, subfolder=subfolder, batch_size=batch_size),
            HuggingFaceDatasetWriter(
                REPO_ID,
                output_filename=f"{tok_name}/${{rank}}.parquet",
                local_working_dir=".datatrove/loader_tok",
                adapter=lambda _, doc: {"id": doc.id, "input_ids": doc.metadata["input_ids"]},
                private=False,
            ),
        ],
        logging_dir=".datatrove/logs/finewebedu_tok",
        tasks=os.cpu_count() - 2,  # type: ignore
    )
    pipe_tokenize.run()

    print("Cleaning up ./.datatrove cache")
    shutil.rmtree(".datatrove", ignore_errors=True)


@app.command()
def download_processed(
    tok: str = "bpe32000minipile", local_dir: str = "./data", cache_dir: str = ".data_cache"
) -> None:
    print(f"Downloading {REPO_ID}/{tok} and saving to {local_dir} (cache in {cache_dir})")
    ds: DatasetDict = load_dataset(REPO_ID, tok, cache_dir=cache_dir, num_proc=os.cpu_count())  # type: ignore

    print(f"Splitting {num_train_docs} docs for training and {num_val_docs} for validation")
    total_size = len(ds["train"])
    ds["validation"] = ds["train"].select(range(num_train_docs, total_size))
    ds["train"] = ds["train"].select(range(num_train_docs))

    out_path = f"{local_dir}/{REPO_ID.split('/')[1]}/{tok}"
    print(f"Saving to {out_path}")
    ds.save_to_disk(out_path, max_shard_size="2GB", num_proc=os.cpu_count())  # type: ignore

    print(f"Cleaning up {cache_dir} cache")
    shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    app()