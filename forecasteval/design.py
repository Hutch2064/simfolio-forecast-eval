"""Plan matched requests before any model runs."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
from math import prod

from .data import Series

REGIMES = ("short", "medium", "long")


def anniversary(day: date, years: int) -> date | None:
    if day.year + years > date.max.year:
        return None
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(year=day.year + years, day=28)  # February 29


def calendar_regime(origin: date, end: date) -> str:
    if end <= origin:
        raise ValueError("a forecast must end after its origin")
    short_end = anniversary(origin, 3)
    if short_end is None or end <= short_end:
        return "short"
    long_start = anniversary(origin, 20)
    return "long" if long_start is not None and end >= long_start else "medium"


@dataclass(frozen=True, slots=True)
class Options:
    min_train: int
    test_start: date | None = None
    test_end: date | None = None
    # None: last observed date in each calendar quarter; an integer: stride in rows.
    origin_step: int | None = None
    origin_cap: int | None = None
    max_horizon: int | None = None
    leads: tuple[int, ...] = (1, 5, 10, 21)

    def __post_init__(self) -> None:
        for key, value, minimum in (
            ("min_train", self.min_train, 2),
            ("origin_step", self.origin_step, 1),
            ("origin_cap", self.origin_cap, 2),
            ("max_horizon", self.max_horizon, 1),
        ):
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{key.replace('_', '-')} must be an integer of at least {minimum}")
        if self.test_start is not None and self.test_end is not None and self.test_start > self.test_end:
            raise ValueError("test-start must be on or before test-end")
        if any(type(h) is not int or h <= 0 for h in self.leads):
            raise ValueError("lead checkpoints must be positive integers")
        object.__setattr__(self, "leads", tuple(sorted(set(self.leads))))


@dataclass(frozen=True, slots=True)
class Request:
    series: Series
    origin_index: int
    end_index: int
    sources: tuple[str, ...]
    weight: float = 0.0

    @property
    def origin(self) -> date:
        return self.series.dates[self.origin_index]

    @property
    def end(self) -> date:
        return self.series.dates[self.end_index]

    @property
    def key(self) -> tuple[str, str, str]:
        return self.series.name, self.origin.isoformat(), self.end.isoformat()

    @property
    def horizon(self) -> int:
        return self.end_index - self.origin_index

    @property
    def regime(self) -> str:
        return calendar_regime(self.origin, self.end)


@dataclass(frozen=True, slots=True)
class Checkpoint:
    lead: int
    origins: int
    included: int
    beyond_available_test_observations: int
    beyond_configured_maximum: int


@dataclass(frozen=True, slots=True)
class Plan:
    requests: tuple[Request, ...]
    unsupported_series: tuple[str, ...]
    checkpoints: tuple[Checkpoint, ...] = ()


def _quarter_ends(dates: tuple[date, ...]) -> tuple[int, ...]:
    quarters: dict[tuple[int, int], int] = {}
    for i, day in enumerate(dates):
        quarters[day.year, (day.month - 1) // 3] = i
    return tuple(quarters.values())


def _thin(origins: list[int], cap: int | None) -> list[int]:
    if cap is None or len(origins) <= cap:
        return origins
    # Evenly cover the date range by eligible-origin rank, including both endpoints.
    return [origins[round(i * (len(origins) - 1) / (cap - 1))] for i in range(cap)]


def _series_requests(series: Series, options: Options) -> tuple[list[Request], dict[int, list[int]]]:
    dates = series.dates
    last = (bisect_right(dates, options.test_end) if options.test_end else len(dates)) - 1
    first_heldout = bisect_left(dates, options.test_start) if options.test_start else 0
    earliest = max(options.min_train - 1, first_heldout - 1)
    if earliest >= last:
        return [], {}

    # The earliest eligible origin allows short records and an explicit test-start boundary.
    if options.origin_step is None:
        ordinary = sorted({earliest, *(i for i in _quarter_ends(dates) if earliest <= i < last)})
    else:
        ordinary = list(range(earliest, last, options.origin_step))
    ordinary = _thin(ordinary, options.origin_cap)

    chosen: dict[tuple[int, int], set[str]] = defaultdict(set)
    origins = set(ordinary)

    def add(origin: int, horizon: int, source: str) -> None:
        if horizon >= 1 and origin + horizon <= last:
            chosen[origin, origin + horizon].add(source)

    def supported(origin: int) -> int:
        available = last - origin
        return available if options.max_horizon is None else min(available, options.max_horizon)

    for origin in ordinary:
        add(origin, min(supported(origin), max(1, (origin + 1) // 4)), "rolling")
    for numer in (1, 2, 3):
        origin = (last + 1) * numer // 4 - 1
        if earliest <= origin < last:
            origins.add(origin)
            add(origin, supported(origin), f"prefix-{numer}/4")
    if options.test_start is not None:
        add(earliest, supported(earliest), "test-start")

    coverage = {h: [len(origins), 0, 0, 0] for h in options.leads}
    for origin in sorted(origins):
        for horizon in options.leads:
            if horizon > last - origin:
                coverage[horizon][2] += 1
            elif options.max_horizon is not None and horizon > options.max_horizon:
                coverage[horizon][3] += 1
            else:
                add(origin, horizon, "checkpoint")
                coverage[horizon][1] += 1
    return [Request(series, origin, end, tuple(sorted(sources)))
            for (origin, end), sources in sorted(chosen.items())], coverage


def _weighted(requests: list[Request]) -> tuple[Request, ...]:
    forks: dict[tuple[object, ...], set[object]] = defaultdict(set)
    nodes: list[tuple[object, ...]] = []
    for r in requests:
        node = (r.regime, *r.series.group, (r.origin_index, r.end_index))
        nodes.append(node)
        for depth, child in enumerate(node):
            forks[node[:depth]].add(child)
    return tuple(replace(r, weight=1 / prod(len(forks[p[:d]]) for d in range(len(p))))
                 for r, p in zip(requests, nodes, strict=True))


def build_plan(series: tuple[Series, ...], options: Options) -> Plan:
    """Equalize weights within the populated, prespecified study hierarchy."""
    if len({target.name for target in series}) != len(series):
        raise ValueError("series names must be unique")
    requests: list[Request] = []
    unsupported = []
    coverage = {h: [0, 0, 0, 0] for h in options.leads}
    for target in sorted(series, key=lambda v: v.name):
        target_requests, target_coverage = _series_requests(target, options)
        for h, counts in target_coverage.items():
            coverage[h] = [a + b for a, b in zip(coverage[h], counts, strict=True)]
        if target_requests:
            requests.extend(target_requests)
        else:
            unsupported.append(target.name)
    if not requests:
        raise ValueError("no eligible requests; check held-out dates and minimum training length")
    return Plan(_weighted(requests), tuple(unsupported), tuple(Checkpoint(lead, *counts) for lead, counts in coverage.items()))
