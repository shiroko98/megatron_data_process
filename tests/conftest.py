import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REAL_DATA_EXCERPT = FIXTURES_DIR / "real_data_excerpt.jsonl"


@pytest.fixture()
def sample_rows():
    return [
        json.loads(line)
        for line in REAL_DATA_EXCERPT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture()
def sample_jsonl(tmp_path: Path, sample_rows) -> Path:
    input_path = tmp_path / "sample.jsonl"
    input_path.write_text(REAL_DATA_EXCERPT.read_text(encoding="utf-8"), encoding="utf-8")
    return input_path


@pytest.fixture()
def multi_file_input_dir(tmp_path: Path) -> Path:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "b.jsonl").write_text(
        json.dumps({"text": "second"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    nested = input_dir / "nested"
    nested.mkdir()
    (nested / "a.jsonl").write_text(
        json.dumps({"text": "first"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (nested / "ignore.txt").write_text("noop", encoding="utf-8")
    return input_dir
