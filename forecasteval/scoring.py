"""The manuscript's single externally evaluated request score."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import fsum, isfinite, sqrt
from statistics import stdev


def _empirical_crps(samples: Sequence[float], observation: float) -> float:
    """CRPS of the issued equally weighted empirical distribution.

    Between adjacent forecast values the CDF is constant. Integrating its
    squared difference from the observation's step CDF is equivalent to
    mean absolute error minus half the self-inclusive pairwise distance.
    This form never subtracts nearly equal large values.
    """
    ordered = sorted(samples)
    total = len(ordered)
    boundary = ordered[0]
    spans = [max(0.0, boundary - observation)]
    below = 0
    for upper in ordered[1:]:
        below += 1
        fraction = below / total
        left = max(0.0, min(upper, observation) - boundary)
        right = max(0.0, upper - max(boundary, observation))
        spans.append(left * fraction ** 2 + right * (1.0 - fraction) ** 2)
        boundary = upper
    spans.append(max(0.0, observation - boundary))
    return fsum(spans)


def _take_exact(values: Iterable[float], count: int, what: str) -> list[float]:
    source = iter(values)
    result: list[float] = []
    for _ in range(count):
        try:
            item = next(source)
            if isinstance(item, (bool, str, bytes)):
                raise TypeError
            value = float(item)
        except StopIteration:
            raise ValueError(f"{what} contains fewer than {count} values") from None
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{what} contains a nonnumeric value") from None
        if not isfinite(value):
            raise ValueError(f"{what} contains a nonfinite value")
        result.append(value)
    extra = object()
    if next(source, extra) is not extra:
        raise ValueError(f"{what} contains more than {count} values")
    return result


def training_scale(training: Sequence[float]) -> float:
    """Sample standard deviation in daily log-return units, training only."""
    if len(training) < 2 or not all(isfinite(x) for x in training):
        raise ValueError("training scale requires at least two finite observations")
    scale = stdev(training)
    if not isfinite(scale) or scale <= 0:
        raise ValueError("training log-return sample standard deviation must be positive and finite")
    return scale


def request_loss(paths: Iterable[Iterable[float]], observed: Sequence[float],
                 sigma: float, members: int) -> float:
    """Mean lead CRPS of cumulative logs, divided by sigma * sqrt(H)."""
    horizon = len(observed)
    if horizon < 1 or members < 1 or not isfinite(sigma) or sigma <= 0:
        raise ValueError("horizon, ensemble size and training scale must be positive")
    if not all(isfinite(x) for x in observed):
        raise ValueError("realized log returns must be finite")
    scale = sigma * sqrt(horizon)
    if not isfinite(scale) or scale <= 0:
        raise ValueError("request standardization scale must be finite and positive")
    columns: list[list[float]] = [[] for _ in range(horizon)]
    iterator = iter(paths)
    for member in range(members):
        try:
            values = _take_exact(next(iterator), horizon, f"path {member + 1}")
        except StopIteration:
            raise ValueError(f"model returned fewer than {members} paths") from None
        endpoint = 0.0
        for column, change in zip(columns, values, strict=True):
            endpoint += change
            if not isfinite(endpoint):
                raise ValueError("cumulative forecast log return is not finite")
            column.append(endpoint)
    extra = object()
    if next(iterator, extra) is not extra:
        raise ValueError(f"model returned more than {members} paths")

    endpoint = 0.0
    lead_losses: list[float] = []
    for column, change in zip(columns, observed, strict=True):
        endpoint += change
        if not isfinite(endpoint):
            raise ValueError("cumulative realized log return is not finite")
        lead_losses.append(_empirical_crps(column, endpoint) / scale)
    score = fsum(x / horizon for x in lead_losses)
    if not isfinite(score):
        raise ValueError("request score is not finite")
    return score
