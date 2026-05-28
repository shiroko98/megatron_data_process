import subprocess
from pathlib import Path

import pytest

import indexed_dataset
import preprocess_data
import my_tokenizer
from my_tokenizer import RWKVTokenizer, _parse_rwkv_vocab_token


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = Path(r"D:\anaconda\envs\model\python.exe")
RWKV_VOCAB = PROJECT_ROOT / "rwkv_vocab_v20240530.txt"
RWKV_VOCAB_NEW = PROJECT_ROOT / "rwkv_vocab_v20250609.txt"
HF_TOKENIZER_PATH = PROJECT_ROOT / "Qwen1.5-14B-Chat"


def test_rwkv_tokenizer_uses_specified_vocab_file() -> None:
    tokenizer = RWKVTokenizer(str(RWKV_VOCAB), vocab_extra_ids=0)

    assert tokenizer.tokenizer.vocab_filepath == str(RWKV_VOCAB)
    assert tokenizer.vocab_size == tokenizer.tokenizer.vocab_size()
    assert tokenizer.vocab_size > tokenizer.eod
    assert tokenizer.tokenize("hello world")


def test_rwkv_new_vocab_file_is_well_formed() -> None:
    ids = []

    with RWKV_VOCAB_NEW.open("r", encoding="utf-8") as handle:
        for line in handle:
            first = line.index(" ")
            last = line.rindex(" ")
            idx = int(line[:first])
            token = _parse_rwkv_vocab_token(line[first:last])
            encoded = token.encode("utf-8") if isinstance(token, str) else token
            declared_length = int(line[last:])

            assert len(encoded) == declared_length
            ids.append(idx)

    assert ids[0] == 1
    assert ids[-1] == 65532
    assert ids == list(range(1, 65533))


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
        "--tokenizer-model",
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


def test_build_tokenizer_hf_prefers_tokenizer_model(monkeypatch) -> None:
    captured = {}
    args = type(
        "Args",
        (),
        {
            "tokenizer_type": "HFTokenizer",
            "tokenizer_model": str(HF_TOKENIZER_PATH),
            "vocab_file": "legacy-fallback-path",
            "vocab_extra_ids": 0,
            "rank": 1,
            "make_vocab_size_divisible_by": 128,
            "tensor_model_parallel_size": 1,
            "padded_vocab_size": None,
        },
    )()

    class FakeHFTokenizer:
        def __init__(self, tokenizer_name_or_path, vocab_extra_ids, **kwargs):
            captured["path"] = tokenizer_name_or_path
            captured["extra_ids"] = vocab_extra_ids
            self.vocab_size = 32

    monkeypatch.setattr(my_tokenizer, "_HFTokenizer", FakeHFTokenizer)

    tokenizer = my_tokenizer.build_tokenizer(args)

    assert isinstance(tokenizer, FakeHFTokenizer)
    assert captured["path"] == str(HF_TOKENIZER_PATH)
    assert captured["extra_ids"] == 0


def test_build_tokenizer_hf_falls_back_to_vocab_file(monkeypatch) -> None:
    captured = {}
    args = type(
        "Args",
        (),
        {
            "tokenizer_type": "HFTokenizer",
            "tokenizer_model": None,
            "vocab_file": str(HF_TOKENIZER_PATH),
            "vocab_extra_ids": 0,
            "rank": 1,
            "make_vocab_size_divisible_by": 128,
            "tensor_model_parallel_size": 1,
            "padded_vocab_size": None,
        },
    )()

    class FakeHFTokenizer:
        def __init__(self, tokenizer_name_or_path, vocab_extra_ids, **kwargs):
            captured["path"] = tokenizer_name_or_path
            self.vocab_size = 32

    monkeypatch.setattr(my_tokenizer, "_HFTokenizer", FakeHFTokenizer)

    my_tokenizer.build_tokenizer(args)

    assert captured["path"] == str(HF_TOKENIZER_PATH)


def test_build_tokenizer_hf_rejects_odd_tokenizer_kwargs() -> None:
    args = type(
        "Args",
        (),
        {
            "tokenizer_type": "HFTokenizer",
            "tokenizer_model": str(HF_TOKENIZER_PATH),
            "vocab_file": None,
            "tokenizer_kwargs": ["use_fast"],
            "vocab_extra_ids": 0,
            "rank": 1,
            "make_vocab_size_divisible_by": 128,
            "tensor_model_parallel_size": 1,
            "padded_vocab_size": None,
        },
    )()

    with pytest.raises(ValueError, match="entered in pairs"):
        my_tokenizer.build_tokenizer(args)


def test_build_tokenizer_hf_passes_tokenizer_kwargs(monkeypatch) -> None:
    captured = {}
    args = type(
        "Args",
        (),
        {
            "tokenizer_type": "HFTokenizer",
            "tokenizer_model": str(HF_TOKENIZER_PATH),
            "vocab_file": None,
            "tokenizer_kwargs": ["use_fast", False, "revision", "main"],
            "vocab_extra_ids": 0,
            "rank": 1,
            "make_vocab_size_divisible_by": 128,
            "tensor_model_parallel_size": 1,
            "padded_vocab_size": None,
        },
    )()

    class FakeHFTokenizer:
        def __init__(self, tokenizer_name_or_path, vocab_extra_ids, **kwargs):
            captured["path"] = tokenizer_name_or_path
            captured["extra_ids"] = vocab_extra_ids
            captured["kwargs"] = kwargs
            self.vocab_size = 32

    monkeypatch.setattr(my_tokenizer, "_HFTokenizer", FakeHFTokenizer)

    my_tokenizer.build_tokenizer(args)

    assert captured["path"] == str(HF_TOKENIZER_PATH)
    assert captured["extra_ids"] == 0
    assert captured["kwargs"] == {"use_fast": False, "revision": "main"}
