import subprocess
from pathlib import Path

import pytest

import indexed_dataset
import preprocess_data
from my_tokenizer import RWKVTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = Path(r"D:\anaconda\envs\model\python.exe")
RWKV_VOCAB = PROJECT_ROOT / "rwkv_vocab_v20240530.txt"
HF_TOKENIZER_PATH = PROJECT_ROOT / "Qwen1.5-14B-Chat"


def test_rwkv_tokenizer_uses_specified_vocab_file() -> None:
    tokenizer = RWKVTokenizer(str(RWKV_VOCAB), vocab_extra_ids=0)

    assert tokenizer.tokenizer.vocab_filepath == str(RWKV_VOCAB)
    assert tokenizer.vocab_size == tokenizer.tokenizer.vocab_size()
    assert tokenizer.vocab_size > tokenizer.eod
    assert tokenizer.tokenize("hello world")


def test_process_data_creates_binidx_outputs(sample_jsonl: Path, sample_rows, tmp_path: Path) -> None:
    output_prefix = tmp_path / "dataset"

    preprocess_data.process_data(
        input=str(sample_jsonl),
        output_prefix=str(output_prefix),
        tokenizer_type="RWKVTokenizer",
        vocab_file=str(RWKV_VOCAB),
        workers=1,
        max_processes=1,
        append_eod=False,
    )

    dataset_prefix = str(output_prefix) + "_text_document"
    dataset = indexed_dataset.IndexedDataset(dataset_prefix)

    assert len(dataset) == len(sample_rows)
    assert dataset[0].size > 0
    assert dataset[len(sample_rows) - 1].size > 0


def test_process_data_merges_single_file_partitions_with_hf_tokenizer(
    sample_jsonl: Path,
    sample_rows,
    tmp_path: Path,
) -> None:
    output_prefix = tmp_path / "partitioned"

    preprocess_data.process_data(
        input=str(sample_jsonl),
        output_prefix=str(output_prefix),
        tokenizer_type="HFTokenizer",
        vocab_file=str(HF_TOKENIZER_PATH),
        workers=2,
        max_processes=1,
        partitions=2,
        merge_partitions=True,
        append_eod=False,
    )

    dataset_prefix = str(output_prefix) + "_text_document"
    dataset = indexed_dataset.IndexedDataset(dataset_prefix)

    assert len(dataset) == len(sample_rows)
    assert (tmp_path / "partitioned_text_document.bin").exists()
    assert (tmp_path / "partitioned_text_document.idx").exists()


def test_cli_entrypoint_honors_arguments(sample_jsonl: Path, tmp_path: Path) -> None:
    output_prefix = tmp_path / "cli_out"
    command = [
        str(PYTHON_EXE),
        str(PROJECT_ROOT / "preprocess_data.py"),
        "--input",
        str(sample_jsonl),
        "--output-prefix",
        str(output_prefix),
        "--tokenizer-type",
        "HFTokenizer",
        "--vocab-file",
        str(HF_TOKENIZER_PATH),
        "--workers",
        "2",
        "--max-processes",
        "1",
        "--partitions",
        "2",
        "--merge-partitions",
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cli_out_text_document.bin").exists()
    assert (tmp_path / "cli_out_text_document.idx").exists()


def _crash_task(_: object) -> None:
    raise RuntimeError("expected child failure")


def test_manage_processes_raises_on_child_failure() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        preprocess_data.manage_processes(_crash_task, [object()], max_processes=1)

    message = str(exc_info.value)
    assert "Child process failed while running" in message
    assert "RuntimeError: expected child failure" in message
    assert "Child traceback:" in message
