Data files omitted. Place analytics competition data here under train/ and supplementary_data.csv.

Derived artifacts:
- `outcome_training.parquet`: built via `python -m analytics.features.outcome_dataset --root <repo> --outputs-dir analytics/outputs/dacs` after generating per-play JSON outputs. This file feeds the calibrated outcome model (DACS probability tuning).
