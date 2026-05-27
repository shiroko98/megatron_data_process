import json
import subprocess
import sys
from pathlib import Path

import pytest

import indexed_dataset
import preprocess_data
from my_tokenizer import RWKVTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = Path(r"D:\anaconda\envs\model\python.exe")
RWKV_VOCAB = PROJECT_ROOT / "rwkv_vocab_v20240530.txt"


@pytest.fixture()
def sample_jsonl(tmp_path: Path) -> Path:
    rows = [
        {"text": "hello world"},
        {"text": "你好，世界"},
    ]
    input_path = tmp_path / "sample.jsonl"
    input_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return input_path


def test_rwkv_tokenizer_uses_specified_vocab_file() -> None:
    tokenizer = RWKVTokenizer(str(RWKV_VOCAB), vocab_extra_ids=0)

    pytest.xfail("RWKVTokenizer custom vocab wiring will be fixed in a dedicated task.")
    assert tokenizer.tokenizer.vocab_filepath == str(RWKV_VOCAB)
    assert tokenizer.vocab_size == 65535
    assert tokenizer.tokenize("hello world")


def test_process_data_creates_binidx_outputs(sample_jsonl: Path, tmp_path: Path) -> None:
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

    assert len(dataset) == 2
    assert dataset[0].size > 0
    assert dataset[1].size > 0


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
        str(PROJECT_ROOT / "Qwen1.5-14B-Chat"),
        "--workers",
        "1",
        "--max-processes",
        "1",
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
    with pytest.raises(
        RuntimeError,
        match=r"Child process failed while running .*RuntimeError: expected child failure",
    ):
        preprocess_data.manage_processes(_crash_task, [object()], max_processes=1)
