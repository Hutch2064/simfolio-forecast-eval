"""Read caller-supplied daily returns without changing their valuation dates."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from math import isfinite, log1p
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True, slots=True)
class Series:
    name: str
    kind: str
    asset_class: str
    family: str
    dates: tuple[date, ...]
    log_returns: tuple[float, ...]

    @property
    def group(self) -> tuple[str, ...]:
        # Constructed portfolios use family, not a constituent asset class.
        if self.kind == "portfolio":
            return self.kind, self.family, self.name
        return self.kind, self.asset_class, self.family, self.name


def iso_date(value: str) -> date:
    """Accept an ISO calendar date, not an ISO week, timestamp or compact date."""
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError("expect an ISO date YYYY-MM-DD") from None
    if parsed.isoformat() != value:
        raise ValueError("expect an ISO date YYYY-MM-DD")
    return parsed


def read_returns(file: TextIO, *, return_kind: str = "simple") -> tuple[Series, ...]:
    """Read date/return with optional series/kind/asset_class/family columns."""
    if return_kind not in {"simple", "log"}:
        raise ValueError("return_kind must be 'simple' or 'log'")
    reader = csv.reader(file, strict=True)
    required = {"date", "return"}
    allowed = required | {"series", "kind", "asset_class", "family"}
    parts: dict[str, tuple[tuple[str, str, str], list[date], list[float]]] = {}
    try:
        header = next(reader, None)
        if header is None:
            raise ValueError("return CSV is empty")
        # utf-8-sig handles the BOM on disk; allow an equivalent text stream.
        if header:
            header[0] = header[0].removeprefix("\ufeff")
        if len(header) != len(set(header)) or not required <= set(header) <= allowed:
            raise ValueError("CSV needs date,return and may also include series,kind,asset_class,family")
        for values in reader:
            if not values:
                continue
            if len(values) != len(header):
                raise ValueError("row has a different number of entries than the header")
            row = dict(zip(header, (item.strip() for item in values), strict=True))
            name = row.get("series", "series")
            if not name:
                raise ValueError("'series' must not be blank when that column exists")
            kind = row.get("kind") or "input"
            if kind not in {"input", "portfolio"}:
                raise ValueError("'kind' must be 'input' or 'portfolio'")
            asset_class = row.get("asset_class", "")
            if kind == "portfolio" and asset_class:
                raise ValueError("leave 'asset_class' blank for constructed portfolios")
            meta = kind, asset_class or "unclassified", row.get("family") or "unclassified"
            day = iso_date(row["date"])
            try:
                value = float(row["return"])
            except ValueError:
                raise ValueError("return must be a finite decimal number") from None
            if not isfinite(value) or (return_kind == "simple" and value <= -1.0):
                raise ValueError("returns must be finite; simple returns must exceed -1")
            series_meta, dates, logs = parts.setdefault(name, (meta, [], []))
            if series_meta != meta:
                raise ValueError("classification changes within a series")
            if dates and day <= dates[-1]:
                raise ValueError("dates must be unique and strictly increasing within each series")
            dates.append(day)
            logs.append(log1p(value) if return_kind == "simple" else value)
    except (csv.Error, ValueError) as exc:
        raise ValueError(f"CSV line {max(1, reader.line_num)}: {exc}") from None
    if not parts:
        raise ValueError("return CSV has no observations")
    return tuple(Series(name, *meta, tuple(dates), tuple(logs))
                 for name, (meta, dates, logs) in sorted(parts.items()))


def load_returns(path: str | Path, *, return_kind: str = "simple") -> tuple[Series, ...]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return read_returns(file, return_kind=return_kind)
