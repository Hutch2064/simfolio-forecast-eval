"""Paired, circular stationary bootstrap on the caller's origin-calendar quarters.

Bring your own data and prespecify the expected length of geometric blocks.
This is not the original study or its automatic block selector. No reference,
expected block length, comparison, decision, or study data is inferred from
model names.
"""
from __future__ import annotations

from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import floor, fsum, isclose, isfinite, log1p
from random import Random
from sys import float_info

from .design import Plan, REGIMES


def _quarter_number(day) -> int:
    return (day.year - 1) * 4 + (day.month - 1) // 3


def _quarter_name(number: int) -> str:
    year, quarter = divmod(number, 4)
    return f"{year + 1:04d}-Q{quarter + 1}"


def _positive_int(x, what: str, minimum: int = 1) -> int:
    if type(x) is not int or x < minimum:
        raise ValueError(f"{what} must be an integer of at least {minimum}")
    return x


def _stationary_counts(n: int, block_length: int, rng: Random) -> list[int]:
    """Counts for n positions in a uniform-start circular stationary bootstrap.

    Independent blocks have geometric length with restart probability 1/L.
    Draws can revisit any quarter, including a previously sampled block start.
    Range additions count consecutive quarters in O(n + number of starts).
    """
    changes = [0] * (n + 1)
    remaining = n
    log_continue = log1p(-1.0 / block_length) if block_length > 1 else 0.0
    while remaining:
        start = rng.randrange(n)
        # Truncate the final block without sampling a needless continuation.
        if block_length == 1 or remaining == 1:
            size = 1
        else:
            log_u = log1p(-rng.random())
            size = remaining if log_u <= (remaining - 1) * log_continue else 1 + int(log_u / log_continue)
        stop = start + size
        changes[start] += 1
        if stop > n:
            changes[0] += 1
            changes[n] -= 1
            stop -= n
        changes[stop] -= 1
        remaining -= size
    count = 0
    for i in range(n):
        count += changes[i]
        changes[i] = count
    changes.pop()
    return changes


def _percentile(ordered: Sequence[float], p: float) -> float:
    """Linear interpolation at zero-based rank (n - 1) * p (type 7)."""
    position = (len(ordered) - 1) * p
    i = floor(position)
    weight = position - i
    if not weight:
        return ordered[i]
    return fsum(((1.0 - weight) * ordered[i], weight * ordered[i + 1]))


def _holm(values: Sequence[float | None]) -> list[float | None]:
    """Unestimable, predeclared hypotheses keep their slots as p=1."""
    adjusted: list[float | None] = [None] * len(values)
    running = 0.0
    order = sorted(range(len(values)), key=lambda i: 1.0 if values[i] is None else values[i])
    for j, i in enumerate(order):
        if values[i] is not None:
            running = min(1.0, max(running, (len(values) - j) * values[i]))
            adjusted[i] = running
    return adjusted


Pair = tuple[str, str]


def _pairs(names: Sequence[str], reference: str | None, comparisons: Sequence[Pair] | None) -> tuple[Pair, ...]:
    if reference is not None and comparisons is not None:
        raise ValueError("provide either a reference or explicit candidate/reference pairs, not both")
    if reference is not None:
        if reference not in names:
            raise ValueError("reference must name a supplied loss vector")
        return tuple((name, reference) for name in sorted(names) if name != reference)
    checked: list[Pair] = []
    if comparisons is not None:
        for pair in comparisons:
            if isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise ValueError("each comparison needs a candidate name and a reference name")
            a, b = pair
            if a not in names or b not in names or a == b or (a, b) in checked:
                raise ValueError("comparison pairs must be unique and name two different supplied sources")
            checked.append((a, b))
    return tuple(sorted(checked))


@dataclass
class _PairData:
    # Each entry is a quarter vector, nested by regime. All numerators move
    # with that regime's resampled mass; no quarter or draw is reweighted to
    # pretend it has the original regime mass.
    difference: list[list[float]]
    reference: list[list[float]]
    wins: list[list[float]]


@dataclass
class _Panel:
    mass: list[list[float]]
    dates: dict
    active: list[list[int]]
    pairs: dict[Pair, _PairData]
    scopes: dict[str, tuple[int, ...]]
    excluded: dict[str, str]


def _prepare(plan: Plan, losses: Mapping[str, Sequence[float]], pairs: Sequence[Pair]) -> _Panel:
    requests = plan.requests
    if not requests:
        raise ValueError("inference needs a nonempty request plan")
    if not losses or any(not isinstance(name, str) or not name.strip() for name in losses):
        raise ValueError("supply one complete loss vector per named forecast")
    for values in losses.values():
        if len(values) != len(requests) or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                              or not isfinite(x) or x < 0 for x in values):
            raise ValueError("expected finite, nonnegative losses aligned to every planned request")
    if (any(not isfinite(q.weight) or q.weight <= 0 for q in requests) or
            not isclose(fsum(q.weight for q in requests), 1.0, rel_tol=1e-12, abs_tol=0)):
        raise ValueError("planned request weights must be positive, finite, and sum to one")
    if len({q.key for q in requests}) != len(requests):
        raise ValueError("planned request identities must be unique")
    numbers = [_quarter_number(q.origin) for q in requests]
    first, last = min(numbers), max(numbers)
    n = last - first + 1
    present = {x - first for x in numbers}
    cells: list[list[list[int]]] = [[[] for _ in range(n)] for _ in REGIMES]
    branches: list[set[str]] = [set() for _ in REGIMES]
    for i, (q, number) in enumerate(zip(requests, numbers, strict=True)):
        g = REGIMES.index(q.regime)
        cells[g][number - first].append(i)
        branches[g].add(q.series.kind)
    mass = [[fsum(requests[i].weight for i in cell) for cell in group] for group in cells]
    active = [[i for i, weight in enumerate(row) if weight > 0] for row in mass]
    excluded: dict[str, str] = {}
    scopes: dict[str, tuple[int, ...]] = {}
    if all("input" in b for b in branches):
        if not all(isclose(fsum(group), 1.0 / len(REGIMES), rel_tol=1e-12) for group in mass):
            raise ValueError("overall inference requires equal planned weight across the three regimes")
        scopes["overall"] = tuple(range(len(REGIMES)))
    else:
        excluded["overall"] = "requires input-series support in each of the three regimes"
    for g, label in enumerate(REGIMES):
        if active[g]:
            scopes[label] = (g,)
        else:
            excluded[label] = "no planned requests in this regime"
    data = {}
    for a, b in pairs:
        x, y = losses[a], losses[b]
        # Mean-score tests use differences in original score units, not percentage
        # reductions. Accumulate paired differences directly to avoid subtracting
        # two already-rounded aggregate means when the losses are nearly equal.
        d = [[fsum(requests[i].weight * (y[i] - x[i]) for i in cell) for cell in group] for group in cells]
        r = [[fsum(requests[i].weight * y[i] for i in cell) for cell in group] for group in cells]
        w = [[fsum(requests[i].weight for i in cell if x[i] < y[i]) for cell in group] for group in cells]
        data[a, b] = _PairData(d, r, w)
    calendar = {"first": _quarter_name(first), "last": _quarter_name(last),
                "quarters": n, "empty_quarters": n - len(present),
                "nonempty_quarters_by_regime": {name: len(active[g]) for g, name in enumerate(REGIMES)},
                "both_branches_in_every_regime": all(b == {"input", "portfolio"} for b in branches),
                "branches_by_regime": {name: sorted(branches[g]) for g, name in enumerate(REGIMES)}}
    return _Panel(mass, calendar, active, data, scopes, excluded)


@dataclass
class _Endpoint:
    candidate: str
    reference: str
    scope: str
    groups: tuple[int, ...]
    kind: str
    estimate: float | None
    contrast: float
    unavailable: str | None
    collect_effect: bool
    effects: array = field(default_factory=lambda: array("d"))
    defined_tests: int = 0
    test_minimum: float = float("inf")
    test_maximum: float = float("-inf")
    exceedances: int = 0


def _effect(difference: float, ref_mean: float) -> float | None:
    if ref_mean == 0.0:
        return None
    value = (difference / ref_mean) * 100.0
    return value if isfinite(value) else None


def _means(numerators: Sequence[float], groups: Sequence[int]) -> float:
    return fsum(numerators[g] / len(groups) for g in groups)


def _endpoints(panel: _Panel) -> list[_Endpoint]:
    total_mass = [fsum(row) for row in panel.mass]
    endpoints = []
    for (a, b), data in panel.pairs.items():
        d, r, w = [[fsum(row) / z if z else 0.0 for row, z in zip(matrix, total_mass, strict=True)]
                   for matrix in (data.difference, data.reference, data.wins)]
        for scope, groups in panel.scopes.items():
            difference, ref_mean, win_share = (_means(values, groups) for values in (d, r, w))
            for kind, point, contrast, matrix in (
                ("mean_loss", _effect(difference, ref_mean), difference, data.difference),
                ("strict_win", win_share, win_share - 0.5, data.wins),
            ):
                unavailable = None
                sufficient_support = all(len(panel.active[g]) >= 2 for g in groups)
                if not sufficient_support:
                    unavailable = "fewer than two nonempty origin quarters in a contributing regime"
                else:
                    # A ratio invariant in every constituent regime is invariant
                    # in every defined draw, even if the regime masses fluctuate.
                    invariant = True
                    for g in groups:
                        rates = [matrix[g][i] / panel.mass[g][i] for i in panel.active[g]]
                        if not all(isclose(rate, rates[0], rel_tol=2e-15, abs_tol=0) for rate in rates):
                            invariant = False
                            break
                    if invariant:
                        unavailable = "test contrast has no between-quarter variation"
                collect_effect = sufficient_support and point is not None and (kind == "mean_loss" or unavailable is None)
                endpoints.append(_Endpoint(a, b, scope, groups, kind, point, contrast, unavailable, collect_effect))
    return endpoints


def _draw(panel: _Panel, counts: Sequence[int], endpoints: Sequence[_Endpoint]) -> None:
    # Use fsum on occupied cells rather than subtracting nearly equal prefix
    # sums: a tiny but positive long-regime mass must not turn into an empty draw.
    # Multiplying all counts by the same constant changes no ratio. Keeping
    # each factor <= 1 also avoids overflow with large, but finite, own losses.
    norm = max(counts)
    selected = [[(i, counts[i] / norm) for i in active if counts[i]] for active in panel.active]
    mass = [fsum(panel.mass[g][i] * c for i, c in items) for g, items in enumerate(selected)]
    per_pair = {}
    for pair, data in panel.pairs.items():
        means = []
        for matrix in (data.difference, data.reference, data.wins):
            means.append([fsum(c * matrix[g][i] for i, c in items) / mass[g] if mass[g] else 0.0
                          for g, items in enumerate(selected)])
        per_pair[pair] = means
    for endpoint in endpoints:
        if endpoint.unavailable and not endpoint.collect_effect:
            continue
        groups = endpoint.groups
        if any(not mass[g] for g in groups):
            # No renormalization across the remaining regimes. Count this as a
            # conservative exceedance, and do not put it in a percentile sample.
            if endpoint.unavailable is None:
                endpoint.exceedances += 1
            continue
        d, r, w = per_pair[endpoint.candidate, endpoint.reference]
        if endpoint.kind == "mean_loss":
            test_value = _means(d, groups)
            value = _effect(test_value, _means(r, groups))
        else:
            value = _means(w, groups)
            test_value = value - 0.5
        if endpoint.collect_effect and value is not None:
            endpoint.effects.append(value)
        if endpoint.unavailable:
            continue
        endpoint.defined_tests += 1
        endpoint.test_minimum = min(endpoint.test_minimum, test_value)
        endpoint.test_maximum = max(endpoint.test_maximum, test_value)
        gap, threshold = abs(test_value - endpoint.contrast), abs(endpoint.contrast)
        # The mathematical comparison includes equality. Treat roundoff-level
        # ties conservatively (e.g. two different quarter compositions with a
        # theoretically zero resampled loss difference).
        if gap >= threshold or isclose(gap, threshold, rel_tol=8 * float_info.epsilon, abs_tol=0):
            endpoint.exceedances += 1


def _run(panel: _Panel, length: int, reps: int, seed: int, primary: bool) -> dict:
    n = panel.dates["quarters"]
    endpoints = _endpoints(panel)
    # Separate streams for each length: reordering a sensitivity check or adding
    # a model to the family does not silently change a pre-existing draw stream.
    rng = Random(f"origin-calendar-quarter:{seed}:{length}")
    if any(e.unavailable is None or e.collect_effect for e in endpoints):
        for _ in range(reps):
            _draw(panel, _stationary_counts(n, length, rng), endpoints)
    reasons = [e.unavailable or ("fewer than two defined bootstrap test draws" if e.defined_tests < 2
                                  else "all defined bootstrap test contrasts are identical" if
                                  isclose(e.test_minimum, e.test_maximum, rel_tol=2e-15, abs_tol=0)
                                  else None) for e in endpoints]
    ps = [(e.exceedances + 1) / (reps + 1) if reason is None else None
          for e, reason in zip(endpoints, reasons, strict=True)]
    family_tail = 0.025 / len(endpoints)
    records = []
    for e, reason, p, adj in zip(endpoints, reasons, ps, _holm(ps), strict=True):
        ordered = sorted(e.effects)
        intervals_ok = len(ordered) >= 2 and e.estimate is not None
        row = {"candidate": e.candidate, "reference": e.reference, "scope": e.scope, "endpoint": e.kind,
               "estimate": e.estimate, "estimate_units": "percent_reduction" if e.kind == "mean_loss" else "win_share",
               "observed_test_contrast": e.contrast, "unavailable_reason": reason,
               "defined_test_draws": None if e.unavailable else e.defined_tests,
               "undefined_test_draws": None if e.unavailable else reps - e.defined_tests,
               "defined_effect_draws": len(ordered) if e.collect_effect else None,
               "undefined_effect_draws": reps - len(ordered) if e.collect_effect else None,
               "conservative_exceedances": None if e.unavailable else e.exceedances,
               "raw_p": p, "holm_adjusted_p": adj,
               "pointwise_95": [_percentile(ordered, .025), _percentile(ordered, .975)] if intervals_ok else None}
        if primary:
            row["bonferroni_95"] = ([_percentile(ordered, family_tail), _percentile(ordered, 1 - family_tail)]
                                    if intervals_ok else None)
        if e.estimate is None and e.kind == "mean_loss":
            row["effect_unavailable_reason"] = "zero reference score or unrepresentable percentage reduction"
        if intervals_ok and ordered[0] == ordered[-1]:
            row["interval_variation_warning"] = "all defined resampled effects are identical; this is not evidence of certainty"
        if intervals_ok and len(ordered) * (family_tail if primary else .025) < 1:
            row["interval_resolution_warning"] = "less than one expected defined draw in an outer percentile tail"
        records.append(row)
    return {"role": "primary" if primary else "sensitivity", "expected_block_length_quarters": length,
            "replications": reps, "monte_carlo_p_resolution": 1 / (reps + 1),
            "record_in_expected_blocks": n / length, "expected_block_starts": 1 + (n - 1) / length,
            "probability_no_restart": (1.0 - 1.0 / length) ** (n - 1),
            "records": records}


def paired_inference(plan: Plan, losses: Mapping[str, Sequence[float]], *, reference: str | None = None,
                     block_length: int, replications: int, seed: int, sensitivity_lengths: Sequence[int] = (),
                     comparisons: Sequence[Pair] | None = None) -> dict:
    """Compute independent, fixed-design comparisons from complete plan-order losses.

    Supply either one named reference (every other named candidate is tested
    against it), or an explicit family of (candidate, reference) pairs. Freeze
    choices and lengths before inspecting outcomes. Overall requires input
    series in each of the three regimes. Weights use the branches in the
    caller's plan; the calendar report identifies missing branches. The
    family is fixed by the plan and choices, not by wins.

    Returns conditional diagnostics, never reject/accept labels. For long blocks
    the calendar can span few mean lengths; no-restart draws reproduce all
    original quarter counts. A percentile interval can therefore be misleadingly
    narrow. A fixed Monte Carlo budget is not the paper's adaptive precision test.
    """
    _positive_int(block_length, "primary expected block length")
    _positive_int(replications, "replications", 2)
    if type(seed) is not int:
        raise ValueError("bootstrap seed must be an integer")
    lengths = tuple(_positive_int(v, "sensitivity expected block length") for v in sensitivity_lengths)
    if len({block_length, *lengths}) != len(lengths) + 1:
        raise ValueError("sensitivity lengths must be distinct and not repeat the primary length")
    pairs = _pairs(tuple(losses), reference, comparisons)
    panel = _prepare(plan, losses, pairs)
    common = {"method": "paired_circular_stationary_bootstrap_origin_calendar_quarters",
              "calendar": panel.dates, "comparisons": [{"candidate": a, "reference": b} for a, b in pairs],
              "excluded_scopes": panel.excluded}
    if not pairs:
        return {**common, "status": "not_applicable", "reason": "an explicit reference and another forecast are required",
                "runs": []}
    runs = [_run(panel, length, replications, seed, i == 0) for i, length in enumerate((block_length, *lengths))]
    estimable = any(e["raw_p"] is not None or e["pointwise_95"] is not None for run in runs for e in run["records"])
    return {**common, "status": "computed" if estimable else "not_estimable", "block_length_selection": "caller_prespecified_not_automatically_selected",
            "bootstrap_seed": seed, "replication_policy": "fixed budget per length; no sequential precision certification",
            "scope": "conditional on the declared models, references, planned support and weights; no model-selection adjustment",
            "family_size": 2 * len(pairs) * len(panel.scopes), "family_scopes": list(panel.scopes),
            "family_definition": "two endpoints per declared comparison per design-supported scope, retaining untestable slots",
            "test": "two-sided centered bootstrap; reference minus candidate score, or strict-win share minus 0.5; plus-one p-values",
            "bands": "linear percentiles on defined effects; primary Bonferroni across this family only; Holm separately at each length",
            "interpretation": "no significance decisions; sensitivity lengths are descriptive and not jointly covered; "
                              "inspect temporal support, no-restart probability, defined draws and Monte Carlo resolution",
            "runs": runs}
