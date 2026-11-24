"""Apply one complete, preweighted experiment to every candidate."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from math import fsum, isfinite
from random import Random

from .design import Options, Plan, REGIMES, Request
from .plugins import Factory, Paths, Training
from .predictions import issued_paths
from .scoring import request_loss, training_scale


def _request_seed(base_seed: int, request: Request) -> int:
    identity = [base_seed, *request.key]
    digest = sha256(dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()).digest()
    return int.from_bytes(digest[:16], "big")


def _buckets(requests: Sequence[Request], key: Callable[[Request], tuple[str, ...]]) -> dict[tuple[str, ...], list[int]]:
    out: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, r in enumerate(requests):
        out[key(r)].append(i)
    return dict(sorted(out.items()))


def design_report(plan: Plan, options: Options, *, show_requests: bool = False) -> dict:
    result = {
        "minimum_training_observations": options.min_train,
        "heldout_start": options.test_start.isoformat() if options.test_start else None,
        "heldout_end": options.test_end.isoformat() if options.test_end else None,
        "origin_step_observations": options.origin_step,
        "rolling_origin_cap_per_series": options.origin_cap,
        "maximum_horizon_observations": options.max_horizon,
        "requested_lead_checkpoints": list(options.leads),
        "request_count": len(plan.requests),
        "series_count": len({r.series.name for r in plan.requests}),
        "unsupported_series": list(plan.unsupported_series),
        "unsupported_regimes": [r for r in REGIMES if not any(q.regime == r for q in plan.requests)],
        "regimes_without_input_series": [r for r in REGIMES if not any(q.regime == r and q.series.kind == "input" for q in plan.requests)],
        "checkpoint_coverage": [
            {
                "horizon_observations": row.lead,
                "eligible_origins": row.origins,
                "included": row.included,
                "skipped_insufficient_test_observations": row.beyond_available_test_observations,
                "skipped_above_configured_maximum": row.beyond_configured_maximum,
            }
            for row in plan.checkpoints
        ],
        "coverage_by_regime": [],
    }
    for regime in REGIMES:
        qs = [q for q in plan.requests if q.regime == regime]
        if qs:
            result["coverage_by_regime"].append({
                "regime": regime,
                "request_count": len(qs),
                "series_count": len({q.series.name for q in qs}),
                "distinct_origin_dates": len({q.origin for q in qs}),
                "distinct_horizon_observations": len({q.horizon for q in qs}),
                "origin_first": min(q.origin for q in qs).isoformat(),
                "origin_last": max(q.origin for q in qs).isoformat(),
                "endpoint_first": min(q.end for q in qs).isoformat(),
                "endpoint_last": max(q.end for q in qs).isoformat(),
                "horizon_min_observations": min(q.horizon for q in qs),
                "horizon_max_observations": max(q.horizon for q in qs),
                "weight": fsum(q.weight for q in qs),
            })
    if show_requests:
        result["requests"] = [{
            "series": q.series.name, "kind": q.series.kind,
            "asset_class": q.series.asset_class if q.series.kind == "input" else None,
            "family": q.series.family, "origin": q.origin.isoformat(), "end": q.end.isoformat(),
            "training_observations": q.origin_index + 1, "horizon_observations": q.horizon,
            "regime": q.regime, "sources": list(q.sources), "weight": q.weight,
        } for q in plan.requests]
    return result


@dataclass(frozen=True, slots=True)
class Candidate:
    """Either a caller plugin or its externally produced forecast ensembles."""

    name: str
    factory: Factory | None = None
    predictions_file: str | None = None

    def __post_init__(self) -> None:
        if not self.name or (self.factory is None) == (self.predictions_file is None):
            raise ValueError("a named candidate needs exactly one plugin or predictions file")

    def paths(self, requests: Iterable[Request], members: int, seed: int) -> Iterator[Paths]:
        if self.predictions_file is not None:
            yield from issued_paths(self.predictions_file, requests)
            return
        factory = self.factory
        assert factory is not None
        for q in requests:
            cut = q.origin_index + 1
            training = Training(q.series.name, q.series.dates[:cut], q.series.log_returns[:cut])
            # Fresh factory state, an immutable training prefix and a deterministic request seed.
            yield factory().forecast(training, q.horizon, members, Random(_request_seed(seed, q)))


def evaluate(plan: Plan, candidates: Sequence[Candidate], *, members: int,
             seed: int, reference: str | None = None, show_losses: bool = False) -> dict:
    if type(members) is not int or members < 1:
        raise ValueError("ensemble size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    names = {candidate.name for candidate in candidates}
    if not candidates or len(names) != len(candidates):
        raise ValueError("provide at least one caller-supplied forecast source and use distinct names")
    if reference is not None and reference not in names:
        raise ValueError("reference must name one of the supplied forecasts")
    qs = plan.requests
    if not qs:
        raise ValueError("evaluate requires a nonempty plan")
    scales: dict[tuple[str, int], float] = {}
    # Preflight every planned training prefix. No failure can change another model's denominator.
    for q in qs:
        key = (q.series.name, q.origin_index)
        if key not in scales:
            try:
                scales[key] = training_scale(q.series.log_returns[:q.origin_index + 1])
            except (ValueError, ArithmeticError) as exc:
                raise ValueError(f"{q.series.name} at {q.origin}: {exc}") from None

    losses: dict[str, list[float]] = {}
    for candidate in candidates:
        vector = []
        iterator = candidate.paths(qs, members, seed)
        try:
            for q in qs:
                try:
                    paths = next(iterator)
                    actual = q.series.log_returns[q.origin_index + 1:q.end_index + 1]
                    loss = request_loss(paths, actual, scales[q.series.name, q.origin_index], members)
                except Exception as exc:
                    raise ValueError(f"{candidate.name} failed for {q.series.name} {q.origin} to {q.end}: {exc}") from None
                vector.append(loss)
            try:
                next(iterator)
            except StopIteration:
                pass
            except Exception as exc:
                raise ValueError(f"{candidate.name}: {exc}") from None
            else:
                raise ValueError(f"{candidate.name}: source supplied excess requests")
        finally:
            iterator.close()
        losses[candidate.name] = vector
    result = summarize_losses(plan, losses, reference=reference)
    if show_losses:
        result["request_losses_in_plan_order"] = losses
    return result


def summarize_losses(plan: Plan, losses: dict[str, list[float]], *, reference: str | None = None) -> dict:
    """Use planned weights to aggregate a complete vector of this score per model."""
    qs = plan.requests
    if not qs or not losses or (reference is not None and reference not in losses):
        raise ValueError("a nonempty, matched loss vector is required")
    if any(len(row) != len(qs) or any(not isfinite(x) or x < 0 for x in row) for row in losses.values()):
        raise ValueError("each forecast source must supply a finite, nonnegative loss for every planned request")

    def scores(indices: Sequence[int]) -> dict:
        mass = fsum(qs[i].weight for i in indices)
        means = {name: fsum((qs[i].weight / mass) * losses[name][i] for i in indices)
                 for name in losses}
        comparison = {"weight": mass, "request_count": len(indices), "loss": means}
        if reference is not None and len(losses) > 1:
            reference_mean = means[reference]
            reductions = {}
            for name, loss in means.items():
                if name == reference:
                    continue
                skill = 100 * (1 - loss / reference_mean) if reference_mean > 0 else None
                reductions[name] = skill if skill is not None and isfinite(skill) else None
            comparison["reduction_percent_vs_reference"] = reductions
            comparison["strict_win_weight_share_vs_reference"] = {
                name: fsum(qs[i].weight for i in indices if losses[name][i] < losses[reference][i]) / mass
                for name in means if name != reference
            }
        return comparison

    scopes: dict[str, Callable[[Request], tuple[str, ...]]] = {
        "regime": lambda q: (q.regime,),
        "kind": lambda q: (q.series.kind,),
        "class": lambda q: (q.series.kind, q.series.asset_class),
        "family": lambda q: q.series.group[:-1],
        "series": lambda q: (q.series.name,),
    }
    subgroups = {}
    for label, key in scopes.items():
        groups = _buckets(qs, key)
        subgroups[label] = [{"group": list(group), **scores(indices)} for group, indices in groups.items()
                            if label != "class" or group[0] == "input"]
    all_regimes = {q.regime for q in qs} == set(REGIMES)
    complete_hierarchy = {q.regime for q in qs if q.series.kind == "input"} == set(REGIMES)
    return {
        "score": "request-standardized-empirical-crps",
        "reference": reference,
        "aggregation_scope": (
            "three-regime" if complete_hierarchy else
            "supported-branches-only" if all_regimes else "supported-regimes-only"
        ),
        "three_regime_aggregate": scores(range(len(qs))) if complete_hierarchy else None,
        "supported_aggregate": None if complete_hierarchy else scores(range(len(qs))),
        "subgroups": subgroups,
    }
