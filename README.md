# Simfolio Forecast Evaluation

An open-source, bring-your-own-data and bring-your-own-model companion to the rolling-origin evaluation in Simfolio’s [Adaptive Simulation of Asset and Portfolio Returns](https://simfol.io/research/simfolio-forecasting-methodology-white-paper.pdf) white paper (Simfolio sign-in required), linked from the [Methodology page](https://simfol.io/methodology).

Score probabilistic forecasts on the same held-out dates using the paper’s request-standardized empirical continuous ranked probability score (CRPS). Lower is better. The harness reports which horizons your data support and can add descriptive diagnostics or paired, origin-quarter bootstrap comparisons. It includes no data, models, benchmarks or historical results and does not reproduce the paper’s historical experiment.

Run from the source directory with Python 3.10 or newer; there are no runtime dependencies. Use the same planning flags for both commands:

```sh
python -m forecasteval plan returns.csv --min-train 500 --show-requests
python -m forecasteval evaluate returns.csv --min-train 500 --members 100 --predictions mine=forecasts.jsonl
```

The returns CSV needs `date,return` with increasing ISO dates and decimal daily simple returns; use `--return-kind log` for log returns. The forecast JSONL needs a `series,origin,end,paths` record for each planned request, in plan order. Each record contains the same number of equally weighted paths, each with one daily log return per held-out observation.

Instead of a JSONL file, pass `--model NAME=MODULE:FACTORY` for your own local Python model. Name a comparator with `--reference NAME`; none is selected for you. See `plan --help` and `evaluate --help` for input options and the plugin interface.

This harness is licensed under the [MIT License](LICENSE).
