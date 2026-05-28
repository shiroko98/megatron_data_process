"""Megatron tokenizers."""

import ast
import hashlib
import os
import tempfile
from abc import ABC
from abc import abstractmethod
from megatron_tokenizer import MegatronTokenizer
from tokenizers import Tokenizer
import pyrwkv_tokenizer
from bert_tokenization import FullTokenizer as FullBertTokenizer
from gpt2_tokenization import GPT2Tokenizer
from typing import List, Optional, Union, Dict
from transformers import AutoTokenizer

def build_tokenizer(args):
    """Initialize tokenizer."""

    if args.tokenizer_type == "HFTokenizer":
        tokenizer_path = getattr(args, "tokenizer_model", None) or args.vocab_file
        if tokenizer_path is None:
            raise ValueError(
                "Missing Hugging Face tokenizer path. Set --tokenizer-model "
                "or provide --vocab-file as a backward-compatible fallback."
            )

        if args.rank == 0:
            print(' > building HFTokenizer tokenizer, '
                    'loading tokenizer from pre-trained model', flush=True)

        hf_tokenizer_kwargs = dict()
        if hasattr(args, "tokenizer_kwargs") and args.tokenizer_kwargs:
            # Accept a flat [key, value, key, value] argument list.
            if len(args.tokenizer_kwargs) % 2 != 0:
                raise ValueError("The token name and token value must be entered in pairs.")

            for i in range(0, len(args.tokenizer_kwargs), 2):
                hf_tokenizer_kwargs[args.tokenizer_kwargs[i]] = \
                    args.tokenizer_kwargs[i + 1]

        tokenizer = _HFTokenizer(
            tokenizer_path,
            vocab_extra_ids=args.vocab_extra_ids,
            **hf_tokenizer_kwargs
        )

    elif args.tokenizer_type.lower() == "RWKVTokenizer".lower():
        if args.rank == 0:
            print(' > building RWKVTokenizer tokenizer, '
                    'loading tokenizer from vocab file', flush=True)
        assert args.vocab_file is not None
        tokenizer = RWKVTokenizer(args.vocab_file, vocab_extra_ids=args.vocab_extra_ids)

    else:
        raise NotImplementedError('{} tokenizer is not implemented.'.format(args.tokenizer_type))

    # Add vocab size (if not already set from a checkpoint).
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer

class RWKVTokenizer():
    """RWKV Tokenizer"""
    def __init__(self, vocab_file, vocab_extra_ids):
        self.vocab_file = vocab_file
        vocab_entries, needs_backend_compat = _load_rwkv_vocab_entries(vocab_file)
        backend_vocab_file = _prepare_backend_compatible_rwkv_vocab_file(
            vocab_file,
            vocab_entries,
            needs_backend_compat,
        )

        try:
            self.tokenizer = pyrwkv_tokenizer.RWKVTokenizer(vocab_filepath=backend_vocab_file)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            raise RuntimeError(
                "Failed to initialize pyrwkv_tokenizer with vocab file "
                f"{backend_vocab_file!r}. The vocab may use a backend-incompatible "
                "literal format."
            ) from exc

        self.backend_vocab_file = backend_vocab_file
        self.uses_compat_vocab = backend_vocab_file != os.path.abspath(vocab_file)
        self._vocab_size = self.tokenizer.vocab_size()
        self.eod_id = 65532
        self._idx2token = {idx: token_bytes for idx, token_bytes in vocab_entries}


    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def vocab(self):
        return self._idx2token

    @property
    def inv_vocab(self):
        return {v: k for k, v in self._idx2token.items()}
    
    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def pad(self):
        return -1

    @property
    def eod(self):
        return self.eod_id

    @property
    def mask(self):
        return -1

    @property
    def additional_special_tokens_ids(self):
        return None


def _parse_rwkv_vocab_token(token_literal):
    value = ast.literal_eval(token_literal.strip())
    if not isinstance(value, (str, bytes)):
        raise ValueError(f"Unsupported RWKV vocab token literal: {token_literal!r}")
    return value


def _load_rwkv_vocab_entries(vocab_file):
    entries = []
    needs_backend_compat = False

    with open(vocab_file, "r", encoding="utf-8") as handle:
        for line in handle:
            first_space = line.index(" ")
            last_space = line.rindex(" ")
            idx = int(line[:first_space])
            token_literal = line[first_space:last_space].strip()
            token = _parse_rwkv_vocab_token(token_literal)
            token_bytes = token.encode("utf-8") if isinstance(token, str) else token
            assert isinstance(token_bytes, bytes)
            assert len(token_bytes) == int(line[last_space:])

            if token_literal != _serialize_rwkv_vocab_token_for_backend(token_bytes):
                needs_backend_compat = True

            entries.append((idx, token_bytes))

    return entries, needs_backend_compat


def _serialize_rwkv_vocab_token_for_backend(token_bytes):
    try:
        decoded = token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decoded = None

    if decoded is not None and decoded.encode("utf-8") == token_bytes:
        return repr(decoded)

    hex_bytes = "".join(f"\\x{byte:02x}" for byte in token_bytes)
    return f"b'{hex_bytes}'"


def _prepare_backend_compatible_rwkv_vocab_file(vocab_file, vocab_entries, needs_backend_compat):
    source_vocab_path = os.path.abspath(vocab_file)
    if not needs_backend_compat:
        return source_vocab_path

    digest = hashlib.sha256()
    for idx, token_bytes in vocab_entries:
        digest.update(f"{idx} ".encode("ascii"))
        digest.update(token_bytes)
        digest.update(b"\n")

    cache_dir = os.path.dirname(os.path.abspath(__file__))
    compat_vocab_path = os.path.join(
        cache_dir,
        f"rwkv_vocab_compat_{digest.hexdigest()[:16]}.txt",
    )
    if os.path.exists(compat_vocab_path):
        return compat_vocab_path

    lines = [
        f"{idx} {_serialize_rwkv_vocab_token_for_backend(token_bytes)} {len(token_bytes)}"
        for idx, token_bytes in vocab_entries
    ]
    fd, temp_path = tempfile.mkstemp(
        prefix="rwkv_vocab_compat_",
        suffix=".txt",
        dir=cache_dir,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines))
            handle.write("\n")
        os.replace(temp_path, compat_vocab_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return compat_vocab_path

class _HFTokenizer(MegatronTokenizer):
    """_HFTokenizer for Hf Pretrained model loading."""

    def __init__(self, tokenizer_name_or_path, vocab_extra_ids, **kwargs):
        name = tokenizer_name_or_path
        super().__init__(name)
        hf_tokenizer_kwargs = kwargs
        if vocab_extra_ids > 0:
            hf_tokenizer_kwargs["additional_special_tokens"] = [f"<extra_id_{_id}>" for _id in range(vocab_extra_ids)]

        hf_tokenizer_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **hf_tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.encoder = self.tokenizer.get_vocab()
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self):
        return len(self.tokenizer)  # vocab_size doesn't contain additional tokens

    @property
    def vocab(self):
        return {
            **{special_token: self.tokenizer.convert_tokens_to_ids(special_token)
               for special_token in self.tokenizer.additional_special_tokens},
            **self.tokenizer.vocab,
        }

    @property
    def inv_vocab(self):
        return {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eos

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def cls(self):
        candidate = self.tokenizer.cls_token_id
        return self._check_token_candidate(candidate)

    @property
    def sep(self):
        candidate = self.tokenizer.sep_token_id
        return self._check_token_candidate(candidate)

    @property
    def pad(self):
        candidate = self.tokenizer.pad_token_id

        # just use eos_token_id if pad_token_id is not available, it is reasonable
        # maybe add a new token, and resize embedding layer is better
        if candidate is None:
            candidate = self.tokenizer.eos_token_id
        return self._check_token_candidate(candidate)

    @property
    def mask(self):
        candidate = self.tokenizer.mask_token_id
        return self._check_token_candidate(candidate)

    @property
    def bos(self):
        raise NotImplementedError("Missing <bos>")

    @property
    def eos(self):
        candidate = self.tokenizer.eos_token_id
        return self._check_token_candidate(candidate)

    @property
    def additional_special_tokens_ids(self):
        """ All the additional special tokens you may want to use (list of strings)."""
        return self.tokenizer.additional_special_tokens_ids

    @staticmethod
    def _check_token_candidate(candidate):
        if candidate is None:
            raise AttributeError("Token doesn't exist")
        return candidate

def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * \
        args.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after
