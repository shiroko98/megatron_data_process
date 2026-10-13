# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import traceback
import hashlib
from contextlib import ExitStack, contextmanager
from collections import Counter, deque
import tempfile
import signal
from itertools import islice
import torch
import multiprocessing
from queue import Empty
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from my_tokenizer import build_tokenizer
import indexed_dataset
import logging

ERROR_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "error.log",
)
logging.basicConfig(filename=ERROR_LOG_PATH, level=logging.ERROR, encoding="utf-8")

# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
if nltk_available:
    class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

        _period_context_fmt = r"""
            \S*                          # some word material
            %(SentEndChars)s             # a potential sentence ending
            \s*                       #  <-- THIS is what I changed
            (?=(?P<after_tok>
                %(NonWord)s              # either other punctuation
                |
                (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
            ))"""
else:
    class CustomLanguageVars(object):
        pass

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text #返回原text


def _get_json_key_value(data, key):
    value = data.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, (str, list)) or (
        isinstance(value, list) and not all(isinstance(item, str) for item in value)
    ):
        raise ValueError(f"Field {key!r} must be a string or a list of strings")
    return value


class Encoder(object):
    initialization_error = None
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.initialization_error = None
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                raise RuntimeError("NLTK is required for sentence splitting")
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(os.environ.get("NLTK_DATA"), "tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"nltk:{library}"
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        if not isinstance(data, dict):
            raise ValueError("Each JSONL row must be a JSON object")
        output = {}
        for key in self.args.json_keys:
            text = _get_json_key_value(data, key)
            max_len = 1000000
            texts = text if isinstance(text, list) else [text]
            tokens_list = [Encoder.splitter.tokenize(item[i:i+max_len])
                           for item in texts for i in range(0, len(item), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial] # 输出是{'text': ['《王的中殿之凤舞之阙》是连载于红袖添香网的网络小说，作者是流莫星。 都市言情。都市言情 他是朝鲜的王,她是他的妃,在一次的相遇他们在一起,意味可以这样幸福下去,没想到一次...他,从此他便的更冷酷。。。。。多年后,即时再相遇,俩人也成了陌生人。。。。。敬请关注']}
        return json.dumps(output, ensure_ascii=False), len(json_line.encode('utf-8'))

    def encode(self, json_line):
        try:
            data = json.loads(json_line)
        except json.JSONDecodeError as e:
            # 记录错误信息（可选）
            logging.error(f"JSON decode error: {e} - Line: {json_line}")
            return None, None, len(json_line.encode('utf-8'))
        if not isinstance(data, dict):
            raise ValueError("Each JSONL row must be a JSON object")
        if getattr(Encoder, "initialization_error", None):
            raise RuntimeError(Encoder.initialization_error)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = _get_json_key_value(data, key)
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line.encode('utf-8'))

    def worker_initializer(self):
        # Pool otherwise continually replaces workers whose initializer fails.
        Encoder.initialization_error = None
        try:
            self.initializer()
        except Exception:
            Encoder.initialization_error = traceback.format_exc()

    def worker_split(self, json_line):
        if Encoder.initialization_error:
            raise RuntimeError(Encoder.initialization_error)
        return self.split(json_line)


def bounded_documents(pool, function, source, workers):
    """Bound queued input and completed results while preserving document order."""
    while True:
        batch = list(islice(source, 32 * workers * 4))
        if not batch:
            return
        yield from pool.imap(function, batch, 32)


class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name # _ss就是sentence_split后的输出file name
        print("Opening", input_file_name)
        encoder = Encoder(self.args)
        with open(input_file_name, 'r', encoding='utf-8') as fin, \
                open(output_file_name, 'w', encoding='utf-8') as fout:
            pool = multiprocessing.Pool(self.workers, initializer=encoder.worker_initializer)
            try:
                split_docs = bounded_documents(pool, encoder.worker_split, fin, self.workers)
                proc_start = time.time()
                total_bytes_processed = 0
                for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
                    total_bytes_processed += bytes_processed
                    fout.write(doc + "\n")
                    self.print_processing_stats(i, proc_start, total_bytes_processed)
                pool.close()
            except BaseException:
                pool.terminate()
                raise
            finally:
                pool.join()

        if self.args.partitions != 1:
            os.remove(input_file_name)  # 删除输入文件
            print(f"Deleted: {input_file_name}")


    # def process_json_file(self, file_name): # 如果是多分区split，则将其先分成n个partition，然后每个partition处理成_ss，然后将_ss再encode
    #     input_file_name, output_prefix = file_name
    #     print("Opening", input_file_name)
    #     fin = open(input_file_name, 'r', encoding='utf-8')

    #     startup_start = time.time()
    #     encoder = Encoder(self.args)
    #     tokenizer = build_tokenizer(self.args) # 在每个独立的进程内，再启用self.workers个(args.workers//args.partitions)进程池进行并行
    #     pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
    #     encoded_docs = pool.imap(encoder.encode, fin, 32)
    #     pool.close()

    #     level = "document"
    #     if self.args.split_sentences:
    #         level = "sentence"

    #     output_bin_files = {}
    #     output_idx_files = {}
    #     builders = {}

    #     for key in self.args.json_keys:
    #         output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
    #                                                       key, level)
    #         output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
    #                                                       key, level)
    #         builders[key] = indexed_dataset.IndexedDatasetBuilder(
    #             output_bin_files[key],
    #             dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size), # cardinality < 65500
    #             # dtype=np.uint16,
    #         )

    #     startup_end = time.time()
    #     proc_start = time.time()
    #     total_bytes_processed = 0
    #     print("Time to startup:", startup_end - startup_start)
    #     for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
    #         total_bytes_processed += bytes_processed
    #         for key in doc.keys():
    #             builders[key].add_document(doc[key], sentence_lens[key]) # 随时写入bin
    #         self.print_processing_stats(i, proc_start, total_bytes_processed)

    #     pool.join()
    #     fin.close()
    #     for key in builders:
    #         builders[key].finalize(output_idx_files[key]) # 保存idx
    #     # builders[key].finalize(output_idx_files[key]) 
    #     if self.args.partitions != 1 or self.args.split_sentences:
    #         os.remove(input_file_name)  # 删除输入文件
    #         print(f"Deleted: {input_file_name}")
    
    def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        startup_start = time.time()
        encoder = Encoder(self.args)
        with open(input_file_name, 'r', encoding='utf-8') as fin, \
                dataset_builders(self.args, output_prefix) as builders:
            pool = multiprocessing.Pool(self.workers, initializer=encoder.worker_initializer)
            try:
                encoded_docs = bounded_documents(pool, encoder.encode, fin, self.workers)
                proc_start = time.time()
                total_bytes_processed = 0
                skipped = 0
                print("Time to startup:", proc_start - startup_start)
                for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
                    total_bytes_processed += bytes_processed
                    if doc is None:
                        skipped += 1
                        continue
                    for key in doc:
                        builders[key].add_document(doc[key], sentence_lens[key])
                    self.print_processing_stats(i, proc_start, total_bytes_processed)
                if skipped:
                    print(f"Skipped {skipped} invalid JSON rows in {input_file_name}", file=sys.stderr)
                pool.close()
            except BaseException:
                pool.terminate()
                raise
            finally:
                pool.join()
        if self.args.partitions != 1 or self.args.split_sentences:
            os.remove(input_file_name)
            print(f"Deleted: {input_file_name}")

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer', 'Llama2Tokenizer',
                                "RWKVTokenizer", "HFTokenizer",
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help=('Tokenizer model/path. For HFTokenizer, this is '
                             'the preferred Hugging Face tokenizer/model path.'))
    group.add_argument('--vocab-file', type=str, default=None,
                       help=('Path to the vocab file. RWKVTokenizer requires '
                             'this; HFTokenizer accepts it as a backward-'
                             'compatible fallback path.'))
    group.add_argument('--vocab-size', default=786,
                       help='size of vocab for use with NullTokenizer')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'This is the total tokenizer-worker budget shared '
                             'across concurrently running file tasks.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--max-processes', type=int, default=6,
                        help='Maximum number of concurrent file processes')
    group.add_argument('--merge-partitions', action='store_true',
                        help='Merge partition files into a single output.')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    intermediate_dir = get_intermediate_dir(args)
    os.makedirs(intermediate_dir, exist_ok=True)
    input_file_name = os.path.join(intermediate_dir, f"temp_{file_id}.jsonl")
    sentence_split_file = os.path.join(intermediate_dir, f"temp_ss_{file_id}.jsonl")
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def get_intermediate_dir(args):
    return os.path.abspath(args.output_prefix) + "_intermediate"


def get_sentence_split_file(args, input_file):
    intermediate_dir = get_intermediate_dir(args)
    input_file_abs = os.path.abspath(input_file)

    if os.path.isdir(args.input):
        relative_input = os.path.relpath(input_file_abs, os.path.abspath(args.input))
    else:
        relative_input = os.path.basename(input_file_abs)

    file_name, extension = os.path.splitext(relative_input)
    sentence_split_file = os.path.join(intermediate_dir, file_name + "_ss" + extension)
    os.makedirs(os.path.dirname(sentence_split_file), exist_ok=True)
    return sentence_split_file


def remove_empty_dirs(path):
    if not os.path.isdir(path):
        return

    for entry in os.listdir(path):
        child_path = os.path.join(path, entry)
        if os.path.isdir(child_path):
            remove_empty_dirs(child_path)

    if not os.listdir(path):
        os.rmdir(path)


def check_files_exist(in_ss_out_names, key, num_partitions=None):
    if num_partitions is None:
        num_partitions = len(in_ss_out_names)

    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def resolve_runtime_process_settings(workers, max_processes, task_count):
    if workers <= 0 or max_processes <= 0:
        raise ValueError("workers and max_processes must be positive")
    task_count = max(1, task_count)
    concurrent_file_processes = max(1, min(max_processes, task_count, workers))
    workers_per_process = max(1, workers // concurrent_file_processes)
    return concurrent_file_processes, workers_per_process


def get_tokenizer_vocab_size(args):
    cached_vocab_size = getattr(args, "tokenizer_vocab_size", None)
    if cached_vocab_size is None:
        cached_vocab_size = build_tokenizer(args).vocab_size
        args.tokenizer_vocab_size = cached_vocab_size
    return cached_vocab_size


@contextmanager
def dataset_builders(args, output_prefix):
    """Publish bin/idx only after every field has finished successfully."""
    level = 'sentence' if args.split_sentences else 'document'
    targets = {key: f'{output_prefix}_{key}_{level}' for key in args.json_keys}
    for prefix in targets.values():
        for suffix in ('.bin', '.idx'):
            if os.path.exists(prefix + suffix):
                raise FileExistsError(f"Output already exists: {prefix + suffix}")
    parent = os.path.dirname(os.path.abspath(output_prefix))
    os.makedirs(parent, exist_ok=True)
    dtype = indexed_dataset.DType.optimal_dtype(get_tokenizer_vocab_size(args))
    with tempfile.TemporaryDirectory(prefix='.preprocess-', dir=parent) as temporary:
        builders = {}
        published = []
        try:
            for key in targets:
                builders[key] = indexed_dataset.IndexedDatasetBuilder(
                    os.path.join(temporary, key + '.bin'), dtype=dtype)
            yield builders
            for key, builder in builders.items():
                builder.finalize(os.path.join(temporary, key + '.idx'))
            for key, target in targets.items():
                for suffix in ('.bin', '.idx'):
                    # Link refuses concurrent overwrites; staging is on the same filesystem.
                    os.link(os.path.join(temporary, key + suffix), target + suffix)
                    published.append(target + suffix)
        except BaseException:
            for path in published:
                os.remove(path)
            raise
        finally:
            for builder in builders.values():
                builder.data_file.close()

def get_input_files(input_path, exclude_directory=None):
    """根据输入路径返回文件列表，包括多层目录中的文件"""
    if not os.path.exists(input_path):
        raise ValueError("The provided input path does not exist.")

    if os.path.isdir(input_path):
        file_list = []
        for root, dirs, files in os.walk(input_path):
            if exclude_directory:
                dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d))
                           != os.path.abspath(exclude_directory)]
            dirs.sort()
            files.sort()
            for file in files:
                if file.lower().endswith('.jsonl'):  # 使用endswith并忽略大小写
                    file_list.append(os.path.join(root, file))
        return sorted(file_list)
    elif os.path.isfile(input_path) and input_path.lower().endswith('.jsonl'):
        return [input_path]  # 确保单个文件也是.jsonl文件
    else:
        raise ValueError("The input path is a file but does not end with .jsonl.")


def merge_files(args, in_ss_out_names):
    level = "document" if not args.split_sentences else "sentence"
    with dataset_builders(args, args.output_prefix) as builders:
        for key, builder in builders.items():
            for name in in_ss_out_names:
                builder.add_index(f"{name['output_prefix']}_{key}_{level}")
    for key in args.json_keys:
        # 删除分区文件
        for name in in_ss_out_names:
            try:
                partition_bin = f"{name['output_prefix']}_{key}_{level}.bin"
                partition_idx = f"{name['output_prefix']}_{key}_{level}.idx"
                if os.path.exists(partition_bin):
                    os.remove(partition_bin)
                if os.path.exists(partition_idx):
                    os.remove(partition_idx)
                print(f"Deleted: {partition_bin}, {partition_idx}")
            except FileNotFoundError as e:
                print(f"Error deleting files {partition_bin} or {partition_idx}: {str(e)}")

def manage_processes(task_func, task_args_list, max_processes):
    """管理进程的函数，限制最大并发进程数"""
    if max_processes <= 0:
        raise ValueError("max_processes must be positive")
    pending = deque(task_args_list)
    active_processes = []
    error_queue = multiprocessing.Queue()
    reported_errors = {}
    try:
        _manage_processes(task_func, pending, max_processes, active_processes,
                          error_queue, reported_errors)
    finally:
        for p, _ in active_processes:
            if p.is_alive():
                _terminate_task(p)
            p.join()
        error_queue.close()
        error_queue.join_thread()


def _terminate_task(process):
    # A file task owns its tokenizer workers; stop its whole isolated group.
    try:
        if hasattr(os, 'killpg') and os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass


def _manage_processes(task_func, pending, max_processes, active_processes,
                      error_queue, reported_errors):
    while pending or active_processes:
        # 如果有空闲进程槽且还有任务未启动，则启动新的进程
        while pending and len(active_processes) < max_processes:
            task_args = pending.popleft()
            p = multiprocessing.Process(
                target=_run_task_with_error_reporting,
                args=(task_func, task_args, error_queue),
            )
            p.start()
            active_processes.append((p, task_args))

        _drain_error_queue(error_queue, reported_errors)
        
        next_active_processes = []
        failed_process_info = None
        for p, task_args in active_processes:
            if p.is_alive():
                next_active_processes.append((p, task_args))
                continue

            p.join()
            _drain_error_queue(error_queue, reported_errors)
            if p.exitcode != 0 and failed_process_info is None:
                failed_process_info = (p, task_args, reported_errors.get(p.pid))

        if failed_process_info is not None:
            failed_process, failed_task_args, error_info = failed_process_info
            for p, _ in next_active_processes:
                _terminate_task(p)
                p.join()
            raise RuntimeError(_format_child_process_error(
                task_func,
                failed_task_args,
                failed_process,
                error_info,
            ))

        active_processes[:] = next_active_processes

        time.sleep(0.5)  # 稍微等待一下再次检查


def _run_task_with_error_reporting(task_func, task_args, error_queue):
    if hasattr(os, 'setsid') and multiprocessing.parent_process() is not None:
        os.setsid()
    try:
        task_func(task_args)
    except Exception as exc:
        error_queue.put({
            "pid": os.getpid(),
            "task": task_func.__qualname__,
            "task_args": repr(task_args),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


def _drain_error_queue(error_queue, reported_errors):
    while True:
        try:
            error_info = error_queue.get_nowait()
        except Empty:
            break
        reported_errors[error_info["pid"]] = error_info


def _format_child_process_error(task_func, task_args, failed_process, error_info):
    input_file = _extract_input_file_from_task_args(task_args)

    if error_info is None:
        return (
            "Child process failed while running "
            f"{task_func.__qualname__} with args={task_args!r} "
            f"(pid={failed_process.pid}, exitcode={failed_process.exitcode})."
        )

    task_desc = f"file {input_file!r}" if input_file is not None else f"args={task_args!r}"
    message = (
        f"Child process failed while running {error_info['task']} for {task_desc}: "
        f"{error_info['error_type']}: {error_info['error_message']} "
        f"(pid={failed_process.pid}, exitcode={failed_process.exitcode})."
    )
    child_traceback = error_info.get("traceback")
    if child_traceback:
        message += f"\nChild traceback:\n{child_traceback}"
    return message


def _extract_input_file_from_task_args(task_args):
    if isinstance(task_args, str):
        return task_args

    if isinstance(task_args, (tuple, list)) and task_args:
        first_arg = task_args[0]
        if isinstance(first_arg, str):
            return first_arg

    return None

def process_data(input, output_prefix, tokenizer_type, vocab_file, jsonl_keys=["text"], split_sentences=False, keep_newlines=False, tokenizer_model=None, vocab_size=65532, merge_file=None, workers=6, max_processes=6, log_interval=1000, append_eod=True, lang='english', partitions=1, merge_partitions=True, keep_sequential_samples=False):
    if any(value <= 0 for value in (workers, max_processes, partitions, log_interval)):
        raise ValueError("workers, max_processes, partitions and log_interval must be positive")
    if not jsonl_keys or len(set(jsonl_keys)) != len(jsonl_keys):
        raise ValueError("jsonl_keys must be nonempty and unique")
    if any(not key or '/' in key or '\\' in key for key in jsonl_keys):
        raise ValueError("jsonl_keys must not contain path separators")
    # The process pool supplies parallelism; disable HF's inherited thread pool.
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    args = argparse.Namespace(
        input=input,
        output_prefix=output_prefix,
        tokenizer_type=tokenizer_type,
        vocab_file=vocab_file,
        json_keys=jsonl_keys,
        keep_newlines=keep_newlines,
        split_sentences=split_sentences,
        tokenizer_model=tokenizer_model,
        vocab_size=vocab_size,
        merge_file=merge_file,
        workers=workers,
        max_processes=max_processes,
        log_interval=log_interval,
        append_eod=append_eod,
        lang=lang,
        partitions=partitions,
        merge_partitions=merge_partitions,
        keep_sequential_samples=keep_sequential_samples,
        keep_empty=False,
        rank=1,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
        vocab_extra_ids=0
    )
    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")
    
    # supported_tokenizers = [
    #     'BertWordPieceLowerCase', 'BertWordPieceCase',
    #     'GPT2BPETokenizer', 'SentencePieceTokenizer',
    #     'GPTSentencePieceTokenizer', 'Llama2Tokenizer',
    #     'RWKVTokenizer', 'HFTokenizer', 'NullTokenizer', 'PretrainedFromHF'
    # ]
    # if tokenizer_type not in supported_tokenizers:
    #     print("Only support:", ', '.join(supported_tokenizers))

    if args.split_sentences:
        if nltk_available:
            # nltk.data.find('tokenizers/punkt')
            nltk.download("punkt", download_dir=os.environ.get("NLTK_DATA"))
        else:
            raise Exception(
                "nltk library required for sentence splitting is not available.")

    in_ss_out_names = []
    if args.partitions == 1:
        in_file_names = get_input_files(args.input, exclude_directory=get_intermediate_dir(args))
        if not in_file_names:
            raise ValueError("No JSONL files found in the input directory")
        if len(in_file_names) == 1:
            file_names = {
                'partition': in_file_names[0],
                'sentence_split': get_sentence_split_file(args, in_file_names[0]),
                'output_prefix': args.output_prefix}
            in_ss_out_names.append(file_names) # 输入的文件
        else:
            base_counts = Counter(os.path.basename(os.path.splitext(path)[0])
                                  for path in in_file_names)
            for file in in_file_names:
                file_name, extension = os.path.splitext(file)
                sentence_split_file = get_sentence_split_file(args, file)
                output_prefix = args.output_prefix + "_" + os.path.basename(file_name)
                if base_counts[os.path.basename(file_name)] > 1:
                    relative = os.path.relpath(file, args.input)
                    output_prefix += '_' + hashlib.sha256(relative.encode('utf-8')).hexdigest()[:12]
                file_names = {
                'partition': file,
                'sentence_split': sentence_split_file,
                'output_prefix': output_prefix}
                in_ss_out_names.append(file_names) # 输入的文件

    else:
        # in_file_names = glob.glob(args.input) # 写明通配符 # 匹配当前目录及所有子目录中的 txt 文件 txt_files = glob.glob('**/*.txt', recursive=True)
        in_file_names = get_input_files(args.input, exclude_directory=get_intermediate_dir(args))

        # Count total number of lines across .jsonl files 
        if args.keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                with open(filename, "r", encoding='utf-8') as fin:
                    total_sample_count += sum(1 for _ in fin)
            partition_size = max(1, math.ceil(total_sample_count / args.partitions))

        # create .jsonl parition files
        for idx in range(args.partitions):
            in_ss_out_name = get_file_name(args, idx)
            in_ss_out_names.append(in_ss_out_name)


    task_count = len(in_ss_out_names)
    if not in_file_names:
        raise ValueError("No JSONL files found in the input directory")
    level = 'sentence' if split_sentences else 'document'
    prefixes = [name['output_prefix'] for name in in_ss_out_names]
    if merge_partitions and (partitions > 1 or len(in_file_names) > 1):
        prefixes.append(args.output_prefix)
    for prefix in prefixes:
        for key in args.json_keys:
            for suffix in ('.bin', '.idx'):
                path = f'{prefix}_{key}_{level}{suffix}'
                if os.path.exists(path):
                    raise FileExistsError(f"Output already exists: {path}; choose a new output prefix")
    if args.partitions > 1:
        # Recreate scratch data from the current input, never reuse stale contents.
        with ExitStack() as stack:
            partitioned_input_files = [stack.enter_context(open(
                name['partition'], 'w', encoding='utf-8')) for name in in_ss_out_names]
            index = 0
            line_count = 0
            for in_file_name in in_file_names:
                with open(in_file_name, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        if args.keep_sequential_samples:
                            index = min(line_count // partition_size, args.partitions - 1)
                        partitioned_input_files[index].write(line)
                        line_count += 1
                        if not args.keep_sequential_samples:
                            index = (index + 1) % args.partitions
    if args.tokenizer_type.lower() in {"rwkvtokenizer", "hftokenizer"}:
        get_tokenizer_vocab_size(args)
    effective_max_processes, workers_per_process = resolve_runtime_process_settings(
        args.workers,
        args.max_processes,
        task_count,
    )
    partition = Partition(args, workers_per_process)

    # check to see if paritions with split sentences already created
    split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split')

    # split sentences in partition files
    if args.split_sentences:
        task_args_list = [(name['partition'], name['sentence_split']) for name in in_ss_out_names]
        manage_processes(partition.split_sentences, task_args_list, effective_max_processes)
        # processes = []
        # for name in in_ss_out_names:
        #     p = multiprocessing.Process(target=partition.split_sentences,
        #                                 args=((name['partition'], name['sentence_split']),))
        #     p.start()
        #     processes.append(p)

        # for p in processes:
        #     p.join()

        # if args.partitions == 1: # 若是单一文件则结束，若不分区且split，则只到split _ss结束，不encode，感觉不对，所以注释掉
        #     return


    # encode partition files in parallel
    input_key = 'sentence_split' if args.split_sentences else 'partition'
    task_args_list = [(name[input_key], name['output_prefix']) for name in in_ss_out_names]
    manage_processes(partition.process_json_file, task_args_list, effective_max_processes)
    # processes = []
    # input_key = 'sentence_split' if args.split_sentences else 'partition' 
    # for name in in_ss_out_names: # # 每个输入的文件都有一个独立的进程管理
    #     p = multiprocessing.Process(target=partition.process_json_file,
    #                                 args=((name[input_key], name['output_prefix']),))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

    if len(in_file_names) == 1 and partitions == 1:
        remove_empty_dirs(get_intermediate_dir(args))
        return

    # merge bin/idx partitions
    if args.merge_partitions:
        merge_files(args, in_ss_out_names)

    remove_empty_dirs(get_intermediate_dir(args))



def run_from_args(args):
    return process_data(
        input=args.input,
        output_prefix=args.output_prefix,
        tokenizer_type=args.tokenizer_type,
        vocab_file=args.vocab_file,
        jsonl_keys=args.json_keys,
        split_sentences=args.split_sentences,
        keep_newlines=args.keep_newlines,
        tokenizer_model=args.tokenizer_model,
        vocab_size=args.vocab_size,
        merge_file=args.merge_file,
        workers=args.workers,
        max_processes=args.max_processes,
        log_interval=args.log_interval,
        append_eod=args.append_eod,
        lang=args.lang,
        partitions=args.partitions,
        merge_partitions=args.merge_partitions,
        keep_sequential_samples=args.keep_sequential_samples,
    )


def main():
    args = get_args()
    return run_from_args(args)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The function took {elapsed_time} seconds to complete.")

# 单文件单分布
# 多文件单分布
# 单文件多分布
# 多文件多分布
