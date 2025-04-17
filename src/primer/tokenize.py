import os
import shutil
from pathlib import Path

from datatrove.data import DocumentsPipeline
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers.huggingface import ParquetWriter
from datatrove.utils.batching import batched
from huggingface_hub import HfApi
from rich import print
from transformers import AutoTokenizer  # type: ignore


def upload_folder(local_path: str | Path, repo_id: str, path_in_repo: str) -> None:
    assert Path(local_path).exists(), f"Local path {local_path} does not exist"

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, repo_type="dataset")
    print(f"🗂️ Repo created at {repo_id}")

    api.upload_folder(repo_id=repo_id, folder_path=str(local_path), path_in_repo=path_in_repo, repo_type="dataset")
    print(f"✅ Successfully uploaded to {repo_id}/{path_in_repo}")


# Define datatrove `Pipeline` component
class DocumentTokenizer(PipelineStep):
    def __init__(self, tok_path: str, subfolder: str | None = None, batch_size: int = 1000) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, subfolder=subfolder)
        self.batch_size = batch_size

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:  # type: ignore
        for batch in batched(data, self.batch_size):
            with self.track_time(unit="batch"):
                docs = [doc.text for doc in batch]
                encoded_docs: list[list[int]] = self.tokenizer(docs)["input_ids"]  # type: ignore
                for doc, encoded in zip(batch, encoded_docs, strict=True):
                    doc.metadata["input_ids"] = encoded
                    doc.metadata["num_tokens"] = len(encoded)
                    yield doc


# Define the tokenizer function
def tokenize_fn(
    out_path: str | Path,
    source_repo_url: str,
    text_key: str,
    id_key: str,
    tok_path: str,
    subfolder: str | None = None,
    batch_size: int = 1000,
    num_tasks: int | None = None,
    debug: bool = False,
    target_repo_id: str | None = None,
) -> None:
    print(f"Tokenizing {source_repo_url} with {tok_path=}{'/' + subfolder if subfolder else ''} and {batch_size=}")

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    local_working_dir = ".datatrove/working_dir"
    logging_dir = ".datatrove/logs"
    print(f"{out_path=}\n{local_working_dir=}\n{logging_dir=}")

    pipe_tokenize = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                source_repo_url,
                file_progress=True,
                doc_progress=True,
                shuffle_files=False,
                text_key=text_key,
                id_key=id_key,
                limit=100 if debug else -1,
            ),
            DocumentTokenizer(tok_path=tok_path, subfolder=subfolder, batch_size=batch_size),
            ParquetWriter(
                str(out_path),
                output_filename="${rank}.parquet",
                compression="zstd",
                adapter=lambda _, doc: {
                    id_key: doc.id,
                    "input_ids": doc.metadata["input_ids"],
                    "num_tokens": doc.metadata["num_tokens"],
                },
                max_file_size=2 * (2**30),
            ),
        ],
        logging_dir=logging_dir,
        tasks=num_tasks or min(20, os.cpu_count() - 2),  # type: ignore
    )
    pipe_tokenize.run()
    print(f"✅ Successfully tokenized {source_repo_url=} dataset to {out_path=}")

    if target_repo_id is not None:
        path_in_repo = tok_path.replace("/", "__") + ("/" + subfolder if subfolder else "")
        print(f"{target_repo_id=} has been specified. 🆙 Uploading to {target_repo_id}/{path_in_repo}")
        upload_folder(out_path, target_repo_id, path_in_repo)

    print("Cleaning up ./.datatrove cache")
    shutil.rmtree(".datatrove", ignore_errors=True)
