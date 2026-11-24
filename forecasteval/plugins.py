"""A contract for caller-supplied forecasters; no implementation is installed."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from importlib import import_module
from random import Random
from typing import Protocol

Paths = Iterable[Iterable[float]]


@dataclass(frozen=True, slots=True)
class Training:
    """Observed inputs through and including a single origin."""

    series: str
    dates: tuple[date, ...]
    log_returns: tuple[float, ...]


class Forecaster(Protocol):
    def forecast(self, training: Training, horizon: int, members: int, rng: Random) -> Paths:
        """Yield exactly 'members' paths of 'horizon' daily log returns."""
        ...


Factory = Callable[[], Forecaster]


def load_factory(spec: str) -> Factory:
    module, sep, symbol = spec.partition(":")
    if not sep or not symbol.isidentifier() or not all(part.isidentifier() for part in module.split(".")):
        raise ValueError("plugin must be a local module:factory")
    try:
        factory = getattr(import_module(module), symbol)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot load plugin {spec!r}: {exc}") from None
    if not callable(factory):
        raise ValueError(f"plugin factory {spec!r} is not callable")
    return factory
