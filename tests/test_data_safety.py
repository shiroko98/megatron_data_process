import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

import indexed_dataset
import my_tokenizer
import preprocess_data

QWEN = Path('/hikscale/models/Qwen/Qwen3.8-27B')
SMALL_HF = Path(__file__).resolve().parents[1] / 'Qwen1.5-14B-Chat'


def process(source, output, **kwargs):
    return preprocess_data.process_data(
        input=str(source), output_prefix=str(output), tokenizer_type='HFTokenizer',
        vocab_file=None, tokenizer_model=str(SMALL_HF), workers=2, max_processes=2,
        **kwargs)


def read(output):
    data = indexed_dataset.IndexedDataset(str(output) + '_text_document')
    return [data[i].tolist() for i in range(len(data))]


def test_duplicate_basenames_are_not_overwritten_and_merge_matches_tokens(tmp_path):
    source = tmp_path / 'input'
    for folder, text in [('a', 'First document'), ('b', '第二篇文档')]:
        directory = source / folder
        directory.mkdir(parents=True)
        (directory / 'same.jsonl').write_text(json.dumps({'text': text}) + '\n')
    tokenizer = AutoTokenizer.from_pretrained(SMALL_HF)
    output = tmp_path / 'out'
    process(source, output, merge_partitions=True)
    assert read(output) == [tokenizer.encode(t) + [tokenizer.eos_token_id]
                            for t in ['First document', '第二篇文档']]
    assert len(list(source.rglob('same.jsonl'))) == 2


def test_empty_file_in_ordered_partitioning_keeps_exact_document_order(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'a.jsonl').write_text('')
    texts = ['first', 'second', 'third']
    (source / 'b.jsonl').write_text(''.join(json.dumps({'text': t}) + '\n' for t in texts))
    output = tmp_path / 'out'
    process(source, output, partitions=5, keep_sequential_samples=True)
    t = AutoTokenizer.from_pretrained(SMALL_HF)
    assert read(output) == [t.encode(s) + [t.eos_token_id] for s in texts]


def test_empty_ordered_partition_input_is_valid(tmp_path):
    source = tmp_path / 'empty.jsonl'
    source.write_text('')
    output = tmp_path / 'out'
    process(source, output, partitions=2, keep_sequential_samples=True)
    assert read(output) == []


def test_scan_excludes_previous_scratch_data_when_output_is_inside_input(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'data.jsonl').write_text('{"text":"current"}\n')
    output = source / 'out'
    scratch = Path(str(output) + '_intermediate')
    scratch.mkdir()
    (scratch / 'temp_0.jsonl').write_text('{"text":"stale document"}\n')
    process(source, output, partitions=2, keep_sequential_samples=True)
    tokenizer = AutoTokenizer.from_pretrained(SMALL_HF)
    assert read(output) == [tokenizer.encode('current') + [tokenizer.eos_token_id]]


def test_existing_output_is_preserved(tmp_path):
    source = tmp_path / 'in.jsonl'
    source.write_text('{"text":"one"}\n')
    output = tmp_path / 'out'
    process(source, output)
    paths = [Path(str(output) + '_text_document' + s) for s in ('.bin', '.idx')]
    contents = [p.read_bytes() for p in paths]
    source.write_text('{"text":"different"}\n')
    with pytest.raises(FileExistsError):
        process(source, output)
    assert [p.read_bytes() for p in paths] == contents


def test_invalid_field_failure_does_not_publish_partial_dataset(tmp_path):
    source = tmp_path / 'in.jsonl'
    source.write_text('{"text":"one"}\n{"text":42}\n')
    output = tmp_path / 'out'
    with pytest.raises(RuntimeError, match='string or a list'):
        process(source, output)
    assert source.exists()
    assert not list(tmp_path.glob('out*.bin'))
    assert not list(tmp_path.glob('out*.idx'))
    assert not list(tmp_path.glob('.preprocess-*'))


def test_merge_failure_preserves_all_fields_and_partitions(tmp_path):
    args = argparse.Namespace(json_keys=['text', 'title'], split_sentences=False,
                              tokenizer_vocab_size=64, output_prefix=str(tmp_path / 'merged'))
    for i in range(2):
        for field in args.json_keys:
            if i == 1 and field == 'title':
                continue
            p = str(tmp_path / f'part{i}_{field}_document')
            b = indexed_dataset.IndexedDatasetBuilder(p + '.bin', dtype=np.uint16)
            b.add_document([i + 1], [1])
            b.finalize(p + '.idx')
    partitions = list(tmp_path.glob('part*'))
    before = {p: p.read_bytes() for p in partitions}
    with pytest.raises(FileNotFoundError):
        preprocess_data.merge_files(args, [{'output_prefix': str(tmp_path / f'part{i}')} for i in range(2)])
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not list(tmp_path.glob('merged*'))


def test_initializer_failure_returns_instead_of_respawning_workers(monkeypatch, tmp_path):
    if multiprocessing.get_start_method() != 'fork':
        pytest.skip('Injecting child initializer failure requires fork')
    source = tmp_path / 'in.jsonl'
    source.write_text('{"text":"one"}\n')
    def fail(_):
        raise ValueError('test initializer failure')
    monkeypatch.setattr(preprocess_data.Encoder, 'initializer', fail)
    with pytest.raises(RuntimeError, match='test initializer failure'):
        process(source, tmp_path / 'out')
    assert not list(tmp_path.glob('out*.idx'))


def test_hf_sparse_ids_select_dtype_covering_highest_id(monkeypatch):
    class Fake:
        pad_token_id = 0
        def get_vocab(self):
            return {'low': 0, 'high': 100000}
    monkeypatch.setattr(my_tokenizer.AutoTokenizer, 'from_pretrained', lambda *a, **kw: Fake())
    t = my_tokenizer._HFTokenizer('unused', 0)
    assert t.vocab_size == 100001
    assert indexed_dataset.DType.optimal_dtype(t.vocab_size) == np.int32


@pytest.mark.parametrize('filename,expected', [('rwkv_vocab_v20240530.txt', 65535),
                                              ('rwkv_vocab_v20250609.txt', 65532)])
def test_rwkv_eod_is_read_from_the_selected_vocabulary(filename, expected):
    path = Path(preprocess_data.__file__).parent / filename
    tokenizer = my_tokenizer.RWKVTokenizer(str(path), 0)
    assert tokenizer.eod == expected
    assert tokenizer.vocab[tokenizer.eod] in (b'<|endoftext|>', b'<|im_end|>')
    assert tokenizer.vocab_size > max(tokenizer.vocab)


@pytest.mark.skipif(not QWEN.exists(), reason='Local Qwen tokenizer unavailable')
def test_qwen_large_vocabulary_parallel_cli_exact_ids(tmp_path):
    import subprocess
    import sys
    source = tmp_path / 'qwen.jsonl'
    # More than one bounded queue window per partition, with high special IDs.
    texts = ['你好，这是 RWKV 测试。Hello!', '<|im_start|>user\n测试<|im_end|>'] * 150
    source.write_text(''.join(json.dumps({'text': t}, ensure_ascii=False) + '\n' for t in texts), encoding='utf-8')
    output = tmp_path / 'qwen'
    script = Path(preprocess_data.__file__)
    result = subprocess.run([sys.executable, str(script), '--input', str(source),
        '--output-prefix', str(output), '--tokenizer-type', 'HFTokenizer',
        '--tokenizer-model', str(QWEN), '--workers', '2', '--max-processes', '2',
        '--partitions', '2', '--keep-sequential-samples', '--merge-partitions',
        '--append-eod'], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    assert read(output) == [tokenizer.encode(t) + [tokenizer.eos_token_id] for t in texts]
    data = indexed_dataset.IndexedDataset(str(output) + '_text_document')
    assert data[0].dtype == np.int32
    assert max(max(row) for row in read(output)) > 65535


def test_returned_array_remains_valid_after_dataset_is_released(tmp_path):
    import subprocess
    import sys
    prefix = str(tmp_path / 'data')
    builder = indexed_dataset.IndexedDatasetBuilder(prefix + '.bin', dtype=np.int32)
    builder.add_document([100000, 248046], [2])
    builder.finalize(prefix + '.idx')
    # A broken mmap lifetime causes a native crash, so isolate this regression.
    script = '''
import gc, sys
from indexed_dataset import IndexedDataset
dataset = IndexedDataset(sys.argv[1])
tokens = dataset[0]
del dataset
gc.collect()
assert tokens.tolist() == [100000, 248046]
assert IndexedDataset(sys.argv[1])[0].tolist() == [100000, 248046]
'''
    result = subprocess.run([sys.executable, '-c', script, prefix],
                            capture_output=True, text=True, timeout=30,
                            cwd=Path(preprocess_data.__file__).parent)
    assert result.returncode == 0, result.stderr


def test_multimodal_document_defaults_one_mode_per_sequence(tmp_path):
    prefix = str(tmp_path / 'multi')
    builder = indexed_dataset.IndexedDatasetBuilder(prefix + '.bin', dtype=np.int32, multimodal=True)
    builder.add_document([1, 2, 3], [1, 2])
    builder.finalize(prefix + '.idx')
    dataset = indexed_dataset.IndexedDataset(prefix, multimodal=True)
    assert dataset[0][0].tolist() == [1]
    assert dataset[1][0].tolist() == [2, 3]
    assert dataset[0][1] == dataset[1][1] == 0


def test_builder_rejects_inconsistent_lengths_before_writing(tmp_path):
    path = tmp_path / 'bad.bin'
    builder = indexed_dataset.IndexedDatasetBuilder(str(path), dtype=np.int32)
    try:
        with pytest.raises(ValueError, match='all tokens'):
            builder.add_document([1, 2, 3], [1])
        assert builder.data_file.tell() == 0
        assert builder.sequence_lengths == []
    finally:
        builder.data_file.close()


@pytest.mark.parametrize('option', ['workers', 'max_processes', 'partitions', 'log_interval'])
def test_invalid_concurrency_is_rejected_before_work(option, tmp_path):
    kwargs = dict(input='not-needed.jsonl', output_prefix=str(tmp_path / 'out'),
                  tokenizer_type='HFTokenizer', vocab_file=None)
    kwargs[option] = 0
    with pytest.raises(ValueError, match='positive'):
        preprocess_data.process_data(**kwargs)
    assert list(tmp_path.iterdir()) == []
