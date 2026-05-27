import argparse
import builtins
import json
import time
from pathlib import Path

import numpy as np
import pytest

import indexed_dataset
import preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl_rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_args(**overrides):
    values = {
        "input": "input.jsonl",
        "output_prefix": "out",
        "tokenizer_type": "HFTokenizer",
        "vocab_file": str(PROJECT_ROOT / "Qwen1.5-14B-Chat"),
        "json_keys": ["text"],
        "split_sentences": False,
        "keep_newlines": False,
        "tokenizer_model": None,
        "vocab_size": 786,
        "merge_file": None,
        "workers": 1,
        "max_processes": 1,
        "log_interval": 1000,
        "append_eod": False,
        "lang": "english",
        "partitions": 1,
        "merge_partitions": False,
        "keep_sequential_samples": False,
        "keep_empty": False,
        "rank": 1,
        "make_vocab_size_divisible_by": 128,
        "tensor_model_parallel_size": 1,
        "vocab_extra_ids": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_get_file_name_for_directory_input():
    args = _build_args(input="data")

    result = preprocess_data.get_file_name(args, 2)

    assert result["partition"].endswith("temp_2.jsonl")
    assert result["sentence_split"].endswith("temp_ss_2.jsonl")
    assert result["output_prefix"] == "out_2"


def test_check_files_exist_uses_requested_count(tmp_path: Path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text("", encoding="utf-8")

    assert preprocess_data.check_files_exist(
        [{"partition": str(first)}, {"partition": str(second)}],
        "partition",
        1,
    )
    assert not preprocess_data.check_files_exist(
        [{"partition": str(first)}, {"partition": str(second)}],
        "partition",
        2,
    )


def test_check_files_exist_defaults_to_checking_all_entries(tmp_path: Path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text("", encoding="utf-8")

    assert not preprocess_data.check_files_exist(
        [{"partition": str(first)}, {"partition": str(second)}],
        "partition",
    )


def test_get_input_files_directory_is_recursive_and_unsorted_contract(multi_file_input_dir: Path):
    result = preprocess_data.get_input_files(str(multi_file_input_dir))

    assert len(result) == 2
    assert all(path.endswith(".jsonl") for path in result)
    assert any(path.endswith("a.jsonl") for path in result)
    assert any(path.endswith("b.jsonl") for path in result)


def test_get_input_files_ignores_intermediate_jsonl_outputs(multi_file_input_dir: Path):
    (multi_file_input_dir / "temp_0.jsonl").write_text('{"text":"temp"}\n', encoding="utf-8")
    (multi_file_input_dir / "b_ss.jsonl").write_text('{"text":["split"]}\n', encoding="utf-8")
    nested = multi_file_input_dir / "nested"
    (nested / "temp_ss_1.jsonl").write_text('{"text":["temp split"]}\n', encoding="utf-8")

    result = preprocess_data.get_input_files(str(multi_file_input_dir))

    assert len(result) == 2
    assert all("_ss" not in Path(path).name for path in result)
    assert all(not Path(path).name.lower().startswith("temp") for path in result)


def test_get_input_files_single_file(sample_jsonl: Path):
    assert preprocess_data.get_input_files(str(sample_jsonl)) == [str(sample_jsonl)]


def test_get_input_files_rejects_missing_path():
    with pytest.raises(ValueError, match="does not exist"):
        preprocess_data.get_input_files("missing.jsonl")


def test_get_input_files_rejects_non_jsonl_file(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("noop", encoding="utf-8")

    with pytest.raises(ValueError, match="does not end with .jsonl"):
        preprocess_data.get_input_files(str(path))


def test_extract_input_file_from_task_args():
    assert preprocess_data._extract_input_file_from_task_args("a.jsonl") == "a.jsonl"
    assert preprocess_data._extract_input_file_from_task_args(("a.jsonl", "out")) == "a.jsonl"
    assert preprocess_data._extract_input_file_from_task_args((123, "out")) is None


def test_format_child_process_error_with_error_info():
    process = type("P", (), {"pid": 12, "exitcode": 1})()

    message = preprocess_data._format_child_process_error(
        task_func=lambda arg: arg,
        task_args=("input.jsonl", "out"),
        failed_process=process,
        error_info={
            "task": "Partition.process_json_file",
            "error_type": "OSError",
            "error_message": "boom",
            "traceback": "Traceback (most recent call last):\n  ...",
        },
    )

    assert "input.jsonl" in message
    assert "OSError: boom" in message
    assert "Partition.process_json_file" in message
    assert "Child traceback:" in message
    assert "Traceback (most recent call last)" in message


def test_format_child_process_error_without_error_info():
    process = type("P", (), {"pid": 99, "exitcode": 2})()

    message = preprocess_data._format_child_process_error(
        task_func=lambda arg: arg,
        task_args=("input.jsonl", "out"),
        failed_process=process,
        error_info=None,
    )

    assert "args=('input.jsonl', 'out')" in message
    assert "exitcode=2" in message


def test_encoder_encode_handles_invalid_json_logs_and_skips(monkeypatch):
    encoder = preprocess_data.Encoder(_build_args())
    captured = []

    class FakeTokenizer:
        eod = 99

        @staticmethod
        def tokenize(text):
            return [len(text)] if text else []

    preprocess_data.Encoder.tokenizer = FakeTokenizer()
    monkeypatch.setattr(preprocess_data.logging, "error", captured.append)

    doc, lens, processed = encoder.encode("{bad json")

    assert doc is None
    assert lens is None
    assert processed == len("{bad json")
    assert captured


def test_encoder_encode_handles_string_and_list_inputs():
    encoder = preprocess_data.Encoder(_build_args(append_eod=True))

    class FakeTokenizer:
        eod = 7

        @staticmethod
        def tokenize(text):
            return [len(text)] if text else []

    preprocess_data.Encoder.tokenizer = FakeTokenizer()

    doc, lens, processed = encoder.encode(json.dumps({"text": ["ab", "c"]}))

    assert doc["text"] == [2, 1, 7]
    assert lens["text"] == [1, 2]
    assert processed > 0


def test_splitter_identity_returns_original_text():
    splitter = preprocess_data.IdentitySplitter()
    assert splitter.tokenize("abc") == ("abc",)


def test_encoder_initializer_without_sentence_split(monkeypatch):
    encoder = preprocess_data.Encoder(_build_args())
    fake_tokenizer = object()

    monkeypatch.setattr(preprocess_data, "build_tokenizer", lambda args: fake_tokenizer)

    encoder.initializer()

    assert preprocess_data.Encoder.tokenizer is fake_tokenizer
    assert isinstance(preprocess_data.Encoder.splitter, preprocess_data.IdentitySplitter)


def test_encoder_initializer_with_nltk_data_env(monkeypatch):
    args = _build_args(split_sentences=True, keep_newlines=False)
    encoder = preprocess_data.Encoder(args)
    fake_tokenizer = object()
    fake_splitter = type("FakeSplitter", (), {"_params": "params"})()
    captured = {}

    monkeypatch.setattr(preprocess_data, "build_tokenizer", lambda _: fake_tokenizer)
    monkeypatch.setattr(preprocess_data, "nltk_available", True)
    monkeypatch.setenv("NLTK_DATA", str(PROJECT_ROOT))

    def fake_load(url):
        captured["url"] = url
        return fake_splitter

    monkeypatch.setattr(preprocess_data.nltk, "load", fake_load)

    encoder.initializer()

    assert preprocess_data.Encoder.tokenizer is fake_tokenizer
    assert preprocess_data.Encoder.splitter is fake_splitter
    assert captured["url"].startswith("file:")


def test_encoder_initializer_with_keep_newlines(monkeypatch):
    args = _build_args(split_sentences=True, keep_newlines=True)
    encoder = preprocess_data.Encoder(args)
    fake_splitter = type("FakeSplitter", (), {"_params": "params"})()

    monkeypatch.setattr(preprocess_data, "build_tokenizer", lambda _: object())
    monkeypatch.setattr(preprocess_data, "nltk_available", True)
    monkeypatch.delenv("NLTK_DATA", raising=False)
    monkeypatch.setattr(preprocess_data.nltk, "load", lambda _: fake_splitter)

    class FakeSentenceTokenizer:
        def __init__(self, train_text, lang_vars):
            self.train_text = train_text
            self.lang_vars = lang_vars

    monkeypatch.setattr(
        preprocess_data.nltk.tokenize.punkt,
        "PunktSentenceTokenizer",
        FakeSentenceTokenizer,
    )

    encoder.initializer()

    assert preprocess_data.Encoder.splitter.train_text == "params"
    assert isinstance(preprocess_data.Encoder.splitter.lang_vars, preprocess_data.CustomLanguageVars)


def test_encoder_initializer_requires_nltk_when_splitting(monkeypatch):
    encoder = preprocess_data.Encoder(_build_args(split_sentences=True))

    monkeypatch.setattr(preprocess_data, "build_tokenizer", lambda _: object())
    monkeypatch.setattr(preprocess_data, "nltk_available", False)
    monkeypatch.setattr(builtins, "exit", lambda: (_ for _ in ()).throw(SystemExit()))

    with pytest.raises(SystemExit):
        encoder.initializer()


def test_encoder_split_serializes_sentences():
    encoder = preprocess_data.Encoder(_build_args(json_keys=["text"]))

    class FakeSplitter:
        @staticmethod
        def tokenize(text):
            return [text.upper()]

    preprocess_data.Encoder.splitter = FakeSplitter()

    doc, processed = encoder.split(json.dumps({"text": "hello"}))

    assert json.loads(doc) == {"text": ["HELLO"]}
    assert processed > 0


def test_run_from_args_delegates_to_process_data(monkeypatch, sample_jsonl: Path, tmp_path: Path):
    args = _build_args(
        input=str(sample_jsonl),
        output_prefix=str(tmp_path / "out"),
        partitions=2,
        merge_partitions=True,
        keep_sequential_samples=True,
    )
    captured = {}

    def fake_process_data(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(preprocess_data, "process_data", fake_process_data)

    result = preprocess_data.run_from_args(args)

    assert result == "ok"
    assert captured["input"] == str(sample_jsonl)
    assert captured["partitions"] == 2
    assert captured["merge_partitions"] is True
    assert captured["keep_sequential_samples"] is True


def test_main_delegates_to_run_from_args(monkeypatch):
    args = _build_args()
    monkeypatch.setattr(preprocess_data, "get_args", lambda: args)
    monkeypatch.setattr(preprocess_data, "run_from_args", lambda value: ("ok", value))

    result = preprocess_data.main()

    assert result == ("ok", args)


def test_get_args_sets_defaults_and_flags(monkeypatch):
    argv = [
        "preprocess_data.py",
        "--input",
        "input.jsonl",
        "--output-prefix",
        "out",
        "--tokenizer-type",
        "HFTokenizer",
        "--vocab-file",
        str(PROJECT_ROOT / "Qwen1.5-14B-Chat"),
        "--workers",
        "2",
        "--merge-partitions",
    ]
    monkeypatch.setattr(preprocess_data.sys, "argv", argv)

    args = preprocess_data.get_args()

    assert args.input == "input.jsonl"
    assert args.workers == 2
    assert args.merge_partitions is True
    assert args.keep_empty is False
    assert args.vocab_extra_ids == 0


def test_get_args_prints_bert_warning(monkeypatch, capsys):
    argv = [
        "preprocess_data.py",
        "--input",
        "input.jsonl",
        "--output-prefix",
        "out",
        "--tokenizer-type",
        "BertWordPieceCase",
        "--vocab-file",
        "vocab.txt",
        "--workers",
        "1",
    ]
    monkeypatch.setattr(preprocess_data.sys, "argv", argv)

    preprocess_data.get_args()

    assert "Are you sure you don't want to split sentences?" in capsys.readouterr().out


def test_partition_print_processing_stats(capsys):
    partition = preprocess_data.Partition(_build_args(log_interval=1), workers=1)
    partition.print_processing_stats(1, time.time() - 1, 1024 * 1024)

    assert "Processed 1 documents" in capsys.readouterr().err


def test_partition_split_sentences_writes_output_and_removes_input(monkeypatch, tmp_path: Path):
    args = _build_args(partitions=2)
    partition = preprocess_data.Partition(args, workers=1)
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text('{"text":"a"}\n', encoding="utf-8")

    class FakePool:
        def __init__(self, workers, initializer=None):
            if initializer is not None:
                initializer()

        @staticmethod
        def imap(func, fin, chunk_size):
            return iter([('{"text": ["A"]}', 10)])

        @staticmethod
        def close():
            return None

        @staticmethod
        def join():
            return None

    monkeypatch.setattr(preprocess_data, "build_tokenizer", lambda _: object())
    monkeypatch.setattr(preprocess_data.multiprocessing, "Pool", FakePool)

    partition.split_sentences((str(input_path), str(output_path)))

    assert not input_path.exists()
    assert output_path.read_text(encoding="utf-8").strip() == '{"text": ["A"]}'


def test_partition_process_json_file_writes_builders_and_deletes_input(monkeypatch, tmp_path: Path):
    args = _build_args(partitions=2, split_sentences=True, json_keys=["text"])
    partition = preprocess_data.Partition(args, workers=1)
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"text":"a"}\n{"text":"b"}\n', encoding="utf-8")
    output_prefix = str(tmp_path / "out")
    builder_calls = []

    class FakeBuilder:
        def __init__(self, path, dtype):
            self.path = path
            self.dtype = dtype

        def add_document(self, doc, lens):
            builder_calls.append(("add_document", self.path, list(doc), list(lens)))

        def finalize(self, idx_path):
            builder_calls.append(("finalize", self.path, idx_path))

    class FakePool:
        def __init__(self, workers, initializer=None):
            if initializer is not None:
                initializer()

        @staticmethod
        def imap(func, fin, chunk_size):
            return iter([
                ({"text": [1, 2]}, {"text": [2]}, 10),
                (None, None, 5),
            ])

        @staticmethod
        def close():
            return None

        @staticmethod
        def join():
            return None

    class FakeTokenizer:
        vocab_size = 32
        eod = 0

        @staticmethod
        def tokenize(text):
            return [len(text)] if text else []

    monkeypatch.setattr(preprocess_data, "build_tokenizer", lambda _: FakeTokenizer())
    monkeypatch.setattr(preprocess_data.multiprocessing, "Pool", FakePool)
    monkeypatch.setattr(preprocess_data.indexed_dataset, "IndexedDatasetBuilder", FakeBuilder)
    monkeypatch.setattr(preprocess_data.indexed_dataset.DType, "optimal_dtype", lambda _: "dtype")
    monkeypatch.setattr(preprocess_data.Encoder, "initializer", lambda self: setattr(preprocess_data.Encoder, "tokenizer", FakeTokenizer()))
    monkeypatch.setattr(preprocess_data.Encoder, "splitter", preprocess_data.IdentitySplitter())

    partition.process_json_file((str(input_path), output_prefix))

    assert ("add_document", f"{output_prefix}_text_sentence.bin", [1, 2], [2]) in builder_calls
    assert ("finalize", f"{output_prefix}_text_sentence.bin", f"{output_prefix}_text_sentence.idx") in builder_calls
    assert not input_path.exists()


def test_merge_files_merges_and_cleans_partitions(monkeypatch, tmp_path: Path):
    part1 = str(tmp_path / "part1_text_document")
    part2 = str(tmp_path / "part2_text_document")
    for prefix, value in ((part1, [1, 2]), (part2, [3])):
        builder = indexed_dataset.IndexedDatasetBuilder(prefix + ".bin", dtype=np.uint16)
        builder.add_document(value, [len(value)])
        builder.finalize(prefix + ".idx")

    args = _build_args(
        output_prefix=str(tmp_path / "merged"),
        json_keys=["text"],
        split_sentences=False,
    )

    class FakeTokenizer:
        vocab_size = 32

    monkeypatch.setattr(preprocess_data, "build_tokenizer", lambda _: FakeTokenizer())

    preprocess_data.merge_files(
        args,
        [
            {"output_prefix": str(tmp_path / "part1")},
            {"output_prefix": str(tmp_path / "part2")},
        ],
    )

    merged = indexed_dataset.IndexedDataset(str(tmp_path / "merged_text_document"))
    assert len(merged) == 2
    assert not Path(part1 + ".bin").exists()
    assert not Path(part2 + ".idx").exists()


def test_merge_files_ignores_missing_cleanup_files(monkeypatch, tmp_path: Path, capsys):
    args = _build_args(output_prefix=str(tmp_path / "merged"), json_keys=["text"])

    class FakeBuilder:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def add_index(path_prefix):
            return None

        @staticmethod
        def finalize(idx_path):
            return None

    class FakeTokenizer:
        vocab_size = 32

    removals = []

    def fake_remove(path):
        removals.append(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr(preprocess_data, "build_tokenizer", lambda _: FakeTokenizer())
    monkeypatch.setattr(preprocess_data.indexed_dataset, "IndexedDatasetBuilder", FakeBuilder)
    monkeypatch.setattr(preprocess_data.os.path, "exists", lambda _: True)
    monkeypatch.setattr(preprocess_data.os, "remove", fake_remove)

    preprocess_data.merge_files(args, [{"output_prefix": str(tmp_path / "part1")}])

    assert removals
    assert "Error deleting files" in capsys.readouterr().out


def test_run_task_with_error_reporting_success():
    class FakeQueue:
        def put(self, value):
            raise AssertionError("should not be called")

    assert preprocess_data._run_task_with_error_reporting(lambda arg: arg, "ok", FakeQueue()) is None


def test_run_task_with_error_reporting_reports_error():
    captured = []

    class FakeQueue:
        def put(self, value):
            captured.append(value)

    with pytest.raises(ValueError, match="boom"):
        preprocess_data._run_task_with_error_reporting(
            lambda arg: (_ for _ in ()).throw(ValueError("boom")),
            "bad",
            FakeQueue(),
        )

    assert captured[0]["error_type"] == "ValueError"


def test_drain_error_queue_collects_items():
    class FakeQueue:
        def __init__(self, values):
            self.values = list(values)

        def get_nowait(self):
            if not self.values:
                raise preprocess_data.Empty()
            return self.values.pop(0)

    reported = {}
    preprocess_data._drain_error_queue(FakeQueue([{"pid": 1}, {"pid": 2}]), reported)

    assert reported == {1: {"pid": 1}, 2: {"pid": 2}}


def test_process_data_warns_for_bert_without_sentence_split(monkeypatch, capsys, sample_jsonl: Path, tmp_path: Path):
    calls = []
    monkeypatch.setattr(preprocess_data, "manage_processes", lambda task, args, max_processes: calls.append((task, args)))

    preprocess_data.process_data(
        input=str(sample_jsonl),
        output_prefix=str(tmp_path / "out"),
        tokenizer_type="BertWordPieceCase",
        vocab_file="vocab.txt",
        workers=1,
        max_processes=1,
        merge_partitions=False,
    )

    assert "Are you sure you don't want to split sentences?" in capsys.readouterr().out
    assert calls


def test_process_data_requires_nltk_for_sentence_split(sample_jsonl: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(preprocess_data, "nltk_available", False)

    with pytest.raises(Exception, match="nltk library required"):
        preprocess_data.process_data(
            input=str(sample_jsonl),
            output_prefix=str(tmp_path / "out"),
            tokenizer_type="HFTokenizer",
            vocab_file=str(PROJECT_ROOT / "Qwen1.5-14B-Chat"),
            workers=1,
            max_processes=1,
            split_sentences=True,
        )


def test_process_data_multiple_files_partitions_one_uses_per_file_outputs(monkeypatch, multi_file_input_dir: Path, tmp_path: Path):
    pytest.xfail("Windows basename handling for multi-file outputs is a separate pending task.")
    calls = []
    monkeypatch.setattr(preprocess_data, "manage_processes", lambda task, args, max_processes: calls.append(args))

    preprocess_data.process_data(
        input=str(multi_file_input_dir),
        output_prefix=str(tmp_path / "merged"),
        tokenizer_type="HFTokenizer",
        vocab_file=str(PROJECT_ROOT / "Qwen1.5-14B-Chat"),
        workers=1,
        max_processes=1,
        merge_partitions=False,
    )

    assert len(calls) == 1
    task_args = calls[0]
    assert len(task_args) == 2
    assert any(output_prefix.endswith("_a") for _, output_prefix in task_args)
    assert any(output_prefix.endswith("_b") for _, output_prefix in task_args)


def test_process_data_partitioning_keeps_sequential_order(
    monkeypatch,
    sample_jsonl: Path,
    sample_rows,
    tmp_path: Path,
):
    monkeypatch.setattr(preprocess_data, "manage_processes", lambda *args, **kwargs: None)

    preprocess_data.process_data(
        input=str(sample_jsonl),
        output_prefix=str(tmp_path / "out"),
        tokenizer_type="HFTokenizer",
        vocab_file=str(PROJECT_ROOT / "Qwen1.5-14B-Chat"),
        workers=2,
        max_processes=1,
        partitions=2,
        merge_partitions=False,
        keep_sequential_samples=True,
    )

    part0 = sample_jsonl.with_name(sample_jsonl.stem + "_0" + sample_jsonl.suffix)
    part1 = sample_jsonl.with_name(sample_jsonl.stem + "_1" + sample_jsonl.suffix)
    part0_rows = _read_jsonl_rows(part0)
    part1_rows = _read_jsonl_rows(part1)
    partition_size = (len(sample_rows) + 1) // 2

    assert part0_rows == sample_rows[:partition_size]
    assert part1_rows == sample_rows[partition_size:]


def test_process_data_partitioning_round_robins_when_not_sequential(
    monkeypatch,
    sample_jsonl: Path,
    sample_rows,
    tmp_path: Path,
):
    monkeypatch.setattr(preprocess_data, "manage_processes", lambda *args, **kwargs: None)

    preprocess_data.process_data(
        input=str(sample_jsonl),
        output_prefix=str(tmp_path / "out"),
        tokenizer_type="HFTokenizer",
        vocab_file=str(PROJECT_ROOT / "Qwen1.5-14B-Chat"),
        workers=2,
        max_processes=1,
        partitions=2,
        merge_partitions=False,
        keep_sequential_samples=False,
    )

    part0 = sample_jsonl.with_name(sample_jsonl.stem + "_0" + sample_jsonl.suffix)
    part1 = sample_jsonl.with_name(sample_jsonl.stem + "_1" + sample_jsonl.suffix)

    assert _read_jsonl_rows(part0) == sample_rows[0::2]
    assert _read_jsonl_rows(part1) == sample_rows[1::2]


def test_process_data_calls_split_then_encode_when_requested(monkeypatch, sample_jsonl: Path, tmp_path: Path):
    calls = []
    monkeypatch.setattr(preprocess_data.nltk, "download", lambda *args, **kwargs: None)

    def fake_manage(task, args, max_processes):
        calls.append((task.__name__, list(args)))

    monkeypatch.setattr(preprocess_data, "manage_processes", fake_manage)

    preprocess_data.process_data(
        input=str(sample_jsonl),
        output_prefix=str(tmp_path / "out"),
        tokenizer_type="HFTokenizer",
        vocab_file=str(PROJECT_ROOT / "Qwen1.5-14B-Chat"),
        workers=1,
        max_processes=1,
        split_sentences=True,
        merge_partitions=False,
    )

    assert calls[0][0] == "split_sentences"
    assert calls[1][0] == "process_json_file"


def test_process_data_multi_file_partitions_one_rechecks_all_sentence_split_outputs(
    monkeypatch,
    multi_file_input_dir: Path,
    tmp_path: Path,
):
    calls = []
    monkeypatch.setattr(preprocess_data.nltk, "download", lambda *args, **kwargs: None)

    def fake_manage(task, args, max_processes):
        calls.append((task.__name__, list(args)))

    monkeypatch.setattr(preprocess_data, "manage_processes", fake_manage)
    original_exists = preprocess_data.os.path.exists

    def fake_exists(path):
        normalized = str(path).replace("\\", "/")
        if normalized.endswith("/b_ss.jsonl"):
            return True
        if normalized.endswith("/a_ss.jsonl"):
            return False
        return original_exists(path)

    monkeypatch.setattr(preprocess_data.os.path, "exists", fake_exists)

    preprocess_data.process_data(
        input=str(multi_file_input_dir),
        output_prefix=str(tmp_path / "out"),
        tokenizer_type="HFTokenizer",
        vocab_file=str(PROJECT_ROOT / "Qwen1.5-14B-Chat"),
        workers=1,
        max_processes=1,
        split_sentences=True,
        merge_partitions=False,
    )

    assert calls[0][0] == "split_sentences"
    assert len(calls[0][1]) == 2
    assert calls[1][0] == "process_json_file"
