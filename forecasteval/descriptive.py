"""Describe an existing plan and its matched scores without changing the score."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from math import fsum, isclose, isfinite
from typing import TypeVar

from .design import Plan, Request, anniversary

_HORIZON_BINS = (
    "through-1-year",
    "over-1-through-3-years",
    "over-3-through-5-years",
    "over-5-through-10-years",
    "over-10-under-20-years",
    "at-least-20-years",
)
_K = TypeVar("_K")


def _horizon_bin(q: Request) -> str:
    for i, years in enumerate((1, 3, 5, 10)):
        boundary = anniversary(q.origin, years)
        if boundary is None or q.end <= boundary:
            return _HORIZON_BINS[i]
    boundary = anniversary(q.origin, 20)
    return _HORIZON_BINS[4 if boundary is None or q.end < boundary else 5]


def _groups(requests: Sequence[Request], key: Callable[[Request], _K]) -> dict[_K, list[int]]:
    groups: dict[_K, list[int]] = defaultdict(list)
    for i, q in enumerate(requests):
        groups[key(q)].append(i)
    return groups


def _top_share(values: Sequence[float], k: int) -> float | None:
    """Share of a nonnegative total, or undefined when the total is zero."""
    scale = max(values, default=0.0)
    if scale == 0:
        return None
    # Rescale for shares, including contributions of very small magnitude.
    scaled = sorted((value / scale for value in values), reverse=True)
    return fsum(scaled[:k]) / fsum(scaled)


def descriptive_diagnostics(
    plan: Plan,
    losses: Mapping[str, Sequence[float]] | None = None,
    *,
    reference: str | None = None,
) -> dict:
    """Profile the fixed plan, optionally with complete losses in plan order.

    Positive paired loss differences favor the non-reference candidate. These
    summaries count series and describe where weight and loss differences fall;
    they do not treat requests, series, or origin dates as independent draws.
    """
    qs = plan.requests
    weights = tuple(q.weight for q in qs)
    if not weights or any(not isfinite(w) or w <= 0 for w in weights):
        raise ValueError("diagnostics require a nonempty plan with finite positive request weights")
    total_weight = fsum(weights)
    if not isclose(total_weight, 1.0, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("diagnostics require request weights that sum to one")
    if len({q.key for q in qs}) != len(qs) or any(q.origin >= q.end for q in qs):
        raise ValueError("diagnostics require unique dated requests ending after their origins")

    if losses is not None and (
        not losses or any(
            len(row) != len(qs) or any(not isfinite(value) or value < 0 for value in row)
            for row in losses.values()
        )
    ):
        raise ValueError("diagnostics require a finite, nonnegative loss for every planned request and candidate")
    if reference is not None and (losses is None or reference not in losses):
        raise ValueError("reference must name a supplied loss vector")
    reference_losses = losses[reference] if losses is not None and reference is not None else None
    challengers = {name: row for name, row in (losses or {}).items()
                   if reference is not None and name != reference}

    def paired_mean(indices: Sequence[int], row: Sequence[float], mass: float) -> float:
        assert reference_losses is not None
        return fsum((weights[i] / mass) * (reference_losses[i] - row[i]) for i in indices)

    def profile(indices: Sequence[int]) -> dict:
        mass = fsum(weights[i] for i in indices)
        targets = {(qs[i].series.name, qs[i].series.kind) for i in indices}
        result = {
            "supported": bool(indices),
            "weight": mass,
            "request_count": len(indices),
            "series_count": len(targets),
            "input_series_count": sum(kind == "input" for _, kind in targets),
            "portfolio_series_count": sum(kind == "portfolio" for _, kind in targets),
            "distinct_origin_dates": len({qs[i].origin for i in indices}),
            "distinct_endpoint_dates": len({qs[i].end for i in indices}),
            "minimum_horizon_observations": min((qs[i].horizon for i in indices), default=None),
            "maximum_horizon_observations": max((qs[i].horizon for i in indices), default=None),
        }
        if losses is not None:
            means = {name: fsum((weights[i] / mass) * row[i] for i in indices) if indices else None
                     for name, row in losses.items()}
            result["loss"] = means
            if challengers and reference is not None:
                reference_mean = means[reference]
                reductions = {}
                for name in challengers:
                    skill = 100 * (1 - means[name] / reference_mean) if reference_mean else None
                    reductions[name] = skill if skill is not None and isfinite(skill) else None
                result["reduction_percent_vs_reference"] = reductions
                result["paired_loss_difference_vs_reference"] = {
                    name: paired_mean(indices, row, mass) if indices else None
                    for name, row in challengers.items()
                }
        return result

    def year_profile(groups: Mapping[int, Sequence[int]]) -> dict:
        years = sorted(groups)
        return {
            "first_year": years[0],
            "last_year": years[-1],
            "unsupported_year_ranges_inclusive": [
                [left + 1, right - 1] for left, right in zip(years, years[1:]) if right - left > 1
            ],
            "supported_years": [{"year": year, **profile(groups[year])} for year in years],
        }

    horizons = _groups(qs, _horizon_bin)
    calendar = {
        "horizon_definition": (
            "Compare the endpoint with calendar anniversaries of the origin. Upper boundaries at 1, 3, 5 "
            "and 10 anniversaries are inclusive; the 20-year boundary starts the last bin. "
            "February 29 anniversaries fall on February 28 in nonleap years."
        ),
        "year_definition": (
            "Origin year and endpoint year group whole requests separately; neither isolates realized "
            "returns or losses within the year. Gaps between the first and last years lack support."
        ),
        "by_horizon": [{"bin": label, **profile(horizons.get(label, ()))} for label in _HORIZON_BINS],
        "by_origin_year": year_profile(_groups(qs, lambda q: q.origin.year)),
        "by_endpoint_year": year_profile(_groups(qs, lambda q: q.end.year)),
    }

    series = _groups(qs, lambda q: q.series.name)
    breadth = {
        "status": ("available" if challengers else "no-challenger" if reference is not None else "no-reference"),
        "definition": (
            "Each supported series has one vote. Compare the paired loss difference using planned "
            "weights normalized within that series. Positive is a win, zero a tie, and negative a loss. "
            "Ties are in the denominator. Unsupported series do not vote."
        ),
        "supported_series_count": len(series),
        "unsupported_series": list(plan.unsupported_series),
    }
    if challengers:
        series_indices = sorted(series.items())
        series_weight = {name: fsum(weights[i] for i in indices) for name, indices in series_indices}
        comparisons = {}
        for name, row in challengers.items():
            differences = {target: paired_mean(indices, row, series_weight[target])
                           for target, indices in series_indices}
            outcomes = Counter("win" if d > 0 else "loss" if d < 0 else "tie" for d in differences.values())
            comparisons[name] = {
                "winning_series": outcomes["win"],
                "tied_series": outcomes["tie"],
                "losing_series": outcomes["loss"],
                "winning_series_share": outcomes["win"] / len(series),
                "paired_loss_difference_by_series": differences,
            }
        breadth["comparison"] = comparisons

    origin_rows = []
    origins: Mapping[date, Sequence[int]] = _groups(qs, lambda q: q.origin)
    for origin, indices in sorted(origins.items()):
        weight = fsum(weights[i] for i in indices)
        entry = {
            "origin": origin.isoformat(),
            "weight": weight,
            "weight_share": weight / total_weight,
            "request_count": len(indices),
            "series_count": len({qs[i].series.name for i in indices}),
        }
        if challengers:
            entry["contribution_to_overall_loss_difference_vs_reference"] = {
                name: paired_mean(indices, row, total_weight) for name, row in challengers.items()
            }
        origin_rows.append(entry)
    shares = [row["weight_share"] for row in origin_rows]
    concentration = {
        "definition": (
            "Pool all requests with the same origin date across series and endpoints. Each origin "
            "contributes to the signed paired difference in loss units, using the full plan's weight "
            "denominator. Gross absolute contribution sums the magnitudes after pooling by date. "
            "Inverse weight concentration describes allocation, not independent sample size."
        ),
        "distinct_origin_dates": len(origin_rows),
        "top_1_weight_share": _top_share(shares, 1),
        "top_5_weight_share": _top_share(shares, 5),
        "inverse_weight_concentration": 1 / fsum(share ** 2 for share in shares),
        "by_origin": origin_rows,
    }
    if challengers:
        comparisons = {}
        for name in challengers:
            contributions = [row["contribution_to_overall_loss_difference_vs_reference"][name] for row in origin_rows]
            gross = [abs(value) for value in contributions]
            comparisons[name] = {
                "overall_paired_loss_difference": fsum(contributions),
                "gross_absolute_contribution": fsum(gross),
                "top_1_gross_absolute_contribution_share": _top_share(gross, 1),
                "top_5_gross_absolute_contribution_share": _top_share(gross, 5),
            }
        concentration["comparison"] = comparisons
    return {"reference": reference, "calendar_profiles": calendar,
            "series_breadth": breadth, "origin_concentration": concentration}
