"""Read exactly the planned forecast cases from caller-produced JSON Lines."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from json import loads
from pathlib import Path

from .design import Request
from .plugins import Paths


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON number {value}")


def _unique_keys(pairs: Iterable[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON property {key!r}")
        result[key] = value
    return result


def _rows(path: str | Path) -> Iterator[tuple[int, dict]]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = loads(line, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
                if not isinstance(record, dict) or record.keys() != {"series", "origin", "end", "paths"}:
                    raise ValueError("record must contain series, origin, end and paths")
                if not isinstance(record["paths"], list) or not all(isinstance(row, list) for row in record["paths"]):
                    raise ValueError("paths must be an array of arrays")
            except ValueError as exc:
                raise ValueError(f"predictions line {line_number}: {exc}") from None
            yield line_number, record


def issued_paths(path: str | Path, requests: Iterable[Request]) -> Iterator[Paths]:
    # Stream one ensemble at a time: the plan's order also makes duplicates,
    # mistakes and missing cases visible without silently dropping any record.
    rows = _rows(path)
    try:
        for request in requests:
            item = next(rows, None)
            if item is None:
                raise ValueError(f"predictions file ended before request {request.key}")
            line, record = item
            if (record["series"], record["origin"], record["end"]) != request.key:
                raise ValueError(f"predictions line {line}: expected request {request.key} in plan order")
            yield record["paths"]
        if next(rows, None) is not None:
            raise ValueError("predictions file contains cases outside the plan")
    finally:
        rows.close()
