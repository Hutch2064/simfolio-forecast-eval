"""Show dated requests or score caller-supplied forecasts."""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from typing import Sequence

from .data import iso_date, load_returns
from .descriptive import descriptive_diagnostics
from .design import Options, build_plan
from .evaluation import Candidate, design_report, evaluate
from .inference import paired_inference
from .plugins import load_factory


_RETURNS_FORMAT = """Returns input:
  UTF-8 CSV with date,return: strictly increasing ISO YYYY-MM-DD dates per
  series, and finite decimal simple returns > -1 (0.01 means 1%). Use
  --return-kind log for daily log returns. Optional: series, kind (input or
  portfolio), asset_class and family. Set labels consistently within a series.
  Supply already constructed portfolio returns and leave asset_class blank
  for them. Missing dates are not filled.
"""
_PLAN_NOTES = """
Plan output:
  The origin is included in training; the next observed row starts the test.
  The output lists unsupported regimes and skipped named checkpoints. Add
  --show-requests for the dated cases in the order needed by --predictions.
"""
_FORECAST_FORMAT = """
Forecast input:
  Reuse the planning flags. With --predictions, provide one JSON object per
  request from plan --show-requests, in printed order. Keys are exactly
  series, origin, end and paths. Dates must match the plan. paths is an array
  of --members arrays, each of length horizon_observations: daily log
  returns, not prices or cumulative returns.

  With --model NAME=MODULE:FACTORY, the importable factory takes no arguments
  and returns an object with forecast(training, horizon, members, rng) that
  yields the same shape. It is created afresh per request. training exposes
  series, dates and log_returns only through the inclusive origin; rng
  is seeded per request. Only load trusted plugins: they run as Python code.
"""


def _args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecasteval", description=__doc__)
    commands = parser.add_subparsers(required=True, dest="command")
    for command, help_text in (
        ("plan", "inspect supported forecast cases; no model needed"),
        ("evaluate", "score the same cases for one or more caller-supplied forecast sources"),
    ):
        child = commands.add_parser(
            command, help=help_text, formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=_RETURNS_FORMAT + (_PLAN_NOTES if command == "plan" else _FORECAST_FORMAT))
        child.add_argument("returns", help="own dated returns CSV")
        child.add_argument("--return-kind", choices=("simple", "log"), default="simple",
                           help="units in input CSV; forecast paths always contain daily log returns")
        child.add_argument("--min-train", required=True, type=int,
                           help="minimum training observations, including the origin return")
        child.add_argument("--test-start", type=iso_date, help="earliest eligible held-out return date")
        child.add_argument("--test-end", type=iso_date, help="latest eligible held-out return date")
        child.add_argument("--origin-step", type=int,
                           help="rolling origin stride in rows; default: observed quarter ends and first eligible origin")
        child.add_argument("--origin-cap", type=int,
                           help="evenly thin the rolling origins per series, retaining both endpoints")
        child.add_argument("--max-horizon", type=int, help="limit any request to this many held-out observations")
        child.add_argument("--leads", type=int, nargs="*", default=(1, 5, 10, 21),
                           help="explicit observation-count checkpoints; pass flag with no values to omit them")
        child.add_argument("--show-requests", action="store_true", help="include every planned case and weight, and its loss when evaluating")
        child.add_argument("--diagnostics", action="store_true",
                           help="add calendar support, series breadth and origin concentration; descriptive, not inference")
        if command == "evaluate":
            child.add_argument("--members", required=True, type=int, help="number of equally weighted paths per forecast")
            child.add_argument("--seed", default=0, type=int, help="seed base supplied only to caller plugins")
            child.add_argument("--model", metavar="NAME=MODULE:FACTORY", action="append", default=[],
                               help="trusted local caller plugin; repeat to compare")
            child.add_argument("--predictions", metavar="NAME=JSONL", action="append", default=[],
                               help="caller-produced daily-log-return paths for exactly the plan; repeat to compare")
            child.add_argument("--reference", metavar="NAME", help="a supplied comparator (no default)")
            inference = child.add_argument_group("optional paired inference")
            inference.add_argument("--inference", action="store_true",
                                   help="resample geometric blocks of origin calendar quarters for paired comparisons")
            inference.add_argument("--block-quarters", type=int, metavar="L",
                                   help="prespecified primary expected block length in calendar quarters (required with --inference)")
            inference.add_argument("--replications", type=int, metavar="B",
                                   help="fixed number of bootstrap draws per length (required with --inference)")
            inference.add_argument("--inference-seed", type=int, metavar="S",
                                   help="bootstrap seed, separate from the model seed (required with --inference)")
            inference.add_argument("--sensitivity-quarters", type=int, nargs="+", metavar="L",
                                   help="optional distinct expected block lengths, excluding the primary length")
            inference.add_argument("--inference-pair", action="append", metavar="CANDIDATE=REFERENCE",
                                   help="test only named directed pairs; repeatable, instead of a common --reference")
    return parser


def _named(value: str, flag: str) -> tuple[str, str]:
    name, equal, spec = value.partition("=")
    if not equal or not name or "=" in name or name.strip() != name or not spec:
        raise ValueError(f"{flag} requires a nonempty NAME=VALUE")
    return name, spec


def _inference_options(args: argparse.Namespace) -> dict | None:
    """Validate optional settings before invoking expensive caller plugins."""
    controls = (
        ("--block-quarters", args.block_quarters), ("--replications", args.replications),
        ("--inference-seed", args.inference_seed),
    )
    if not args.inference:
        if any(v is not None for _, v in controls) or args.sensitivity_quarters or args.inference_pair:
            raise ValueError("bootstrap settings require --inference")
        return None
    missing = [flag for flag, value in controls if value is None]
    if missing:
        raise ValueError("--inference requires " + ", ".join(missing))
    if args.block_quarters < 1 or args.replications < 2:
        raise ValueError("--block-quarters must be at least 1 and --replications at least 2")
    lengths = tuple(args.sensitivity_quarters or ())
    if any(n < 1 for n in lengths) or len({args.block_quarters, *lengths}) != 1 + len(lengths):
        raise ValueError("sensitivity lengths must be positive, distinct and different from --block-quarters")
    pairs = None
    if args.inference_pair:
        if args.reference is not None:
            raise ValueError("choose either --reference or --inference-pair, not both")
        pairs = tuple(_named(value, "--inference-pair") for value in args.inference_pair)
        names = ({_named(value, "--model")[0] for value in args.model}
                 | {_named(value, "--predictions")[0] for value in args.predictions})
        if len(set(pairs)) != len(pairs) or any(a not in names or b not in names or a == b for a, b in pairs):
            raise ValueError("each --inference-pair must uniquely name two different supplied forecasts")
    return {"block_length": args.block_quarters, "replications": args.replications,
            "seed": args.inference_seed, "sensitivity_lengths": lengths, "comparisons": pairs}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _args()
    args = parser.parse_args(argv)
    try:
        options = Options(args.min_train, args.test_start, args.test_end, args.origin_step,
                          args.origin_cap, args.max_horizon, tuple(args.leads))
        if args.command == "evaluate" and not args.model and not args.predictions:
            raise ValueError("provide --model NAME=MODULE:FACTORY or --predictions NAME=JSONL; no model is included")
        inference = _inference_options(args) if args.command == "evaluate" else None
        dataset = load_returns(args.returns, return_kind=args.return_kind)
        plan = build_plan(dataset, options)
        output = {
            "schema": "forecasteval-v1",
            "input_return_kind": args.return_kind,
            "design": design_report(plan, options, show_requests=args.show_requests),
        }
        matched_losses = None
        if args.command == "evaluate":
            candidates = []
            report_needs_losses = args.diagnostics or inference is not None
            # Keep ordinary plugin prints (including on import) off the JSON stream.
            with redirect_stdout(sys.stderr):
                for argument in args.model:
                    name, plugin = _named(argument, "--model")
                    candidates.append(Candidate(name, factory=load_factory(plugin)))
                for argument in args.predictions:
                    name, file = _named(argument, "--predictions")
                    candidates.append(Candidate(name, predictions_file=file))
                scores = evaluate(plan, candidates, members=args.members, seed=args.seed,
                                  reference=args.reference, show_losses=args.show_requests or report_needs_losses)
            # Reuse exactly the scored vectors; reports never re-run a model or change weights.
            if report_needs_losses:
                matched_losses = scores["request_losses_in_plan_order"]
                if not args.show_requests:
                    del scores["request_losses_in_plan_order"]
            output["evaluation"] = {
                "ensemble_members": args.members, "seed": args.seed,
                **scores,
            }
        if args.diagnostics:
            output["diagnostics"] = descriptive_diagnostics(
                plan, matched_losses, reference=args.reference if args.command == "evaluate" else None)
        if inference is not None:
            assert matched_losses is not None
            output["inference"] = paired_inference(plan, matched_losses, reference=args.reference, **inference)
        rendered = json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False)
    except (OSError, ValueError, ArithmeticError) as exc:
        parser.exit(2, f"forecasteval: {exc}\n")
    sys.stdout.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
