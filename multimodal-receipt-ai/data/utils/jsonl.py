"""JSONL reading and writing utilities."""

import json
import jsonlines
from pathlib import Path
from typing import Iterator, Any, List


def read_jsonl(path: Path | str) -> Iterator[dict]:
    """Read JSONL file line by line."""
    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            yield obj


def write_jsonl(data: List[dict], path: Path | str, mode: str = "w"):
    """Write data to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode=mode) as writer:
        writer.write_all(data)


def append_jsonl(obj: dict, path: Path | str):
    """Append a single object to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="a") as writer:
        writer.write(obj)


def count_jsonl_lines(path: Path | str) -> int:
    """Count lines in JSONL file without loading into memory."""
    count = 0
    with jsonlines.open(path, mode="r") as reader:
        for _ in reader:
            count += 1
    return count


def validate_jsonl_schema(path: Path | str, required_keys: set[str]) -> bool:
    """Validate that all JSONL objects contain required keys."""
    for i, obj in enumerate(read_jsonl(path)):
        if not required_keys.issubset(obj.keys()):
            missing = required_keys - obj.keys()
            raise ValueError(f"Line {i}: missing keys {missing}")
    return True
