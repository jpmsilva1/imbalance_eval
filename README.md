# Imbalanced Time Series Evaluation Library

[![CI](https://github.com/jpmsilva1/imbalance_eval/actions/workflows/ci.yml/badge.svg)](https://github.com/jpmsilva1/imbalance_eval/actions/workflows/ci.yml)

A CLI and Python library that measures **how imbalanced a time series dataset is** — how many of its values are "rare" (unusually extreme) versus "normal."

Point it at a CSV, and it gives you a table like this:

```
 ID Dataset   N Granularity  n_normal  n_rare   IR  %Rare
DS1    temp 731        Daily       658      73  9.0  10.0
```

`N` is the row count, `n_rare` is how many values were flagged extreme, `IR` (imbalance ratio) is `n_normal / n_rare`, and `%Rare` is the percentage of extreme values. This is the same descriptive statistic reported in Table 1 of Moniz, Branco & Torgo (2017), *"Resampling Strategies for Imbalanced Time Series Forecasting"* — this tool implements that paper's relevance-function methodology so you can run it on your own datasets.

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [How the code is organized](#how-the-code-is-organized)
- [Using it on non-time-series data](#using-it-on-non-time-series-data)
- [Usage](#usage)
  - [As a Python library](#as-a-python-library)
  - [CLI: one file](#cli-evaluate-one-csv)
  - [CLI: many files at once](#cli-evaluate-many-datasets-at-once)
  - [All CLI flags](#all-cli-flags)
- [Validation against the paper](#validation-against-the-paper)
- [Known limitations](#known-limitations)
- [Running the tests](#running-the-tests)
- [References](#references)

## Quick start

```bash
pip install ImbalancedLearningRegression pandas numpy scikit-learn
```

Then run it on any CSV — here using the [UCI Bike Sharing dataset](https://doi.org/10.24432/C5W894) as an example:

```bash
python imbalance_eval.py day.csv --target temp --date-col dteday
```

```
 ID Dataset   N Granularity  n_normal  n_rare  IR  %Rare
DS1    temp 721                   721       0 inf    0.0
```

That's it — you have an imbalance reading for the `temp` column (`N=721` because the default pipeline time-delay-embeds the 731-row series with a 10-step window first — see [How it works](#how-it-works)). Keep reading for batch mode (many datasets at once), using it as a library, and what each flag does.

## How it works

1. Take a time series column (e.g. daily temperature, hourly rental counts).
2. Fit a **relevance function** φ(y) that scores every value from 0 (totally normal) to 1 (totally extreme), based on that column's own boxplot statistics (quartiles, IQR, whiskers).
3. Apply a **threshold** (default `0.9`, matching the paper). Any value with φ(y) ≥ threshold counts as "rare."
4. Report `N`, `n_normal`, `n_rare`, `IR` (`n_normal / n_rare`), and `%Rare` (`100 * n_rare / N`).

The relevance function itself isn't reimplemented here — it comes from the [`ImbalancedLearningRegression`](https://pypi.org/project/ImbalancedLearningRegression/) (iblr) package, the Python port of the R `UBL` package the paper's authors used.

The boxplot in step 2 is the **standard (Tukey) boxplot** — not the skew-adjusted variant (Hubert & Vandervieren, 2008) — matching the paper. Both the paper's method (Ribeiro's 2011 thesis → R `UBL`) and `iblr`'s `box_plot_stats()` compute whiskers the classic way: `Q1 - coef*IQR` / `Q3 + coef*IQR` with `coef=1.5` by default, no skew correction.

## How the code is organized

Everything lives in one file, `imbalance_eval.py`: library functions at the top (usable standalone, e.g. from a notebook), a thin `argparse` CLI at the bottom.

| Function | Role |
|---|---|
| `load_csv(path, date_col=None)` | Reads a CSV, optionally sorts by a date column and drops it. |
| `impute_dataframe(df, n_neighbors=5)` | Fills missing values via k-NN, jointly across numeric columns; leaves non-numeric columns untouched; no-ops if nothing's missing. |
| `time_delay_embedding(series, k=10)` | Builds the `[y(t-k), ..., y(t-1), y(t)]` lag table the paper trains on. |
| `compute_imbalance(y, rel_thres=0.9, ...)` | The core calculation: relevance scores in, `{N, n_normal, n_rare, IR, %Rare}` out. |
| `evaluate_datasets(datasets, ...)` | Runs the full pipeline over a list of `{id, name, series, granularity?}` dicts, returns the summary table. |
| `evaluate_csv_files(paths, target, ...)` | Convenience wrapper for Python callers: loads + imputes each path and calls `evaluate_datasets` for you — no manifest file needed. |
| `_run_single` / `_run_batch` / `main` | CLI wiring for the two modes above. |

**One implementation detail worth knowing:** `ImbalancedLearningRegression` 0.0.2 calls a numpy keyword (`interpolation=`) that numpy 2.x removed. `_iblr_numpy_quantile_compat()` patches around it for the duration of that one call only, then restores numpy immediately — so importing this module never silently changes numpy's behavior elsewhere in your process (e.g. other cells in a notebook). Delete it once `iblr` ships a numpy-2-compatible release.

## Using it on non-time-series data

The relevance function doesn't know what a timestamp is. `compute_imbalance()` scores *any* numeric target — insurance claim amounts, manufacturing yields, churn revenue — with no date column and no lag window:

```python
import pandas as pd
from imbalance_eval import compute_imbalance

claims = pd.read_csv("claims.csv")["claim_amount"]
print(compute_imbalance(claims))
```

You get back the same `{N, n_normal, n_rare, IR, %Rare}` dict the time-series path produces. A plain list or numpy array works too — it's converted internally.

**If you use the higher-level helpers on tabular data, pass `embed=False`.** `evaluate_datasets()` and `evaluate_csv_files()` default to `embed=True`, which time-delay-embeds the column first — that's correct for a time series and wrong for unordered rows. It silently drops `k` rows (10 by default) and shifts every statistic:

```python
ds = [{"id": "T1", "name": "claims", "series": claims}]   # 500 rows

evaluate_datasets(ds)                # N=490 -- 10 rows lost to embedding
evaluate_datasets(ds, embed=False)   # N=500 -- what you want for tabular data
```

On the CLI that flag is `--no-embed`.

Two caveats worth knowing before you trust the number:

- **A roughly symmetric target will report few or no rare cases** at the default `--coef 1.5`, since it has no values beyond the Tukey whiskers. This is the same effect documented in [Validation against the paper](#validation-against-the-paper) — tune `--coef` there if your target isn't skewed.
- **This use case isn't validated here.** The DS1–DS8 check in [Validation against the paper](#validation-against-the-paper) validates the *time-series pipeline*. The relevance function itself comes from `iblr` and is used as-is; this project doesn't re-validate it against a tabular regression benchmark.

## Usage

### As a Python library

Use this from a notebook or another script when you want the data in memory rather than piping through the CLI.

**Install directly from GitHub** (e.g. in a Colab cell):

```bash
pip install git+https://github.com/jpmsilva1/imbalance_eval.git
```

```python
import imbalance_eval
print("imbalance_eval is ready to use!")
```

```python
from imbalance_eval import load_csv, impute_dataframe, evaluate_datasets

# 1. Load the CSV into a table (a pandas DataFrame), sorted by date.
#    date_col="dteday" tells it which column is the date -- that column is
#    used to sort the rows in time order, then dropped, since the date
#    itself isn't part of what gets evaluated.
raw = load_csv("day.csv", date_col="dteday")

# 2. Fill in any missing values before evaluating. For each blank cell,
#    this looks at the most similar *other rows* in the table (by their
#    other column values) and fills the gap with their average. If there's
#    no missing data, this is a no-op -- safe to always call.
imputed = impute_dataframe(raw)

# 3. Pick which column(s) you want evaluated. imputed["temp"] pulls out
#    just the temperature column as a single time series; id/name/granularity
#    are just labels for the output table, not used in the calculation.
datasets = [
    {"id": "DS1", "name": "Temperature", "granularity": "Daily", "series": imputed["temp"]},
    {"id": "DS2", "name": "Humidity",    "granularity": "Daily", "series": imputed["hum"]},
]

# 4. Evaluate -- returns a pandas DataFrame, one row per dataset
table = evaluate_datasets(datasets, k=10, rel_thres=0.9)
print(table)
```

Skip embedding to score raw values directly (`evaluate_datasets(datasets, rel_thres=0.9, embed=False)`) — this is also how to reproduce the paper's own Table 1 numbers; see [Validation against the paper](#validation-against-the-paper).

**Evaluating several CSV files at once from Python** — `evaluate_csv_files` does the load-impute-evaluate loop for you, so you don't need to hand-build the `datasets` list above:

```python
from imbalance_eval import evaluate_csv_files

table = evaluate_csv_files(
    ["data/bike_day.csv", "data/bike_hour.csv"],
    target="temp",       # same column name in every file...
    k=10, rel_thres=0.9,
)

# ...or a {path: column} dict when the target column differs per file
paths = ["data/bike_day.csv", "data/river_flow.csv"]
table = evaluate_csv_files(
    paths,
    target={"data/bike_day.csv": "temp", "data/river_flow.csv": "flow"},
    k=10, rel_thres=0.9,
)
```

`id` defaults to each file's name (without extension) and `name` defaults to the target column name — pass `granularity="Daily"` if you want that column filled in, it's optional here too.

### CLI: evaluate one CSV

Use this to check the imbalance of a single column in a single file.

```bash
python imbalance_eval.py path/to/data.csv --target column_name
```

**What happens, step by step:**

1. The CSV is loaded.
2. Missing values (if any) are filled in across all numeric columns.
3. The `column_name` column is extracted.
4. It's time-delay embedded with `k=10` by default — this uses each value's 10 preceding values to build the modeling table the paper trains on. Skip this with `--no-embed` to score the raw column instead.
5. The relevance function runs, rare/normal cases are counted at the threshold (`0.9` by default).
6. A one-row results table prints to your terminal.

**A fuller example**, labeling the output and saving it to a file:

```bash
python imbalance_eval.py day.csv --target temp --date-col dteday \
    --id DS1 --name Temperature --granularity Daily \
    --threshold 0.9 --output results.csv
```

### CLI: evaluate many datasets at once

Use this to build one combined table across several datasets or columns — like reproducing the paper's own Table 1.

**Step 1 — write a manifest CSV** listing what to evaluate. Required columns: `id`, `name`, `path`, `target`. `granularity` is optional (blank if omitted) — it's just a display label, never used in the computation.

```csv
id,name,granularity,path,target
DS1,Temperature,Daily,data/bike_day.csv,temp
DS2,Humidity,Daily,data/bike_day.csv,hum
DS3,Windspeed,Daily,data/bike_day.csv,windspeed
DS4,Count,Daily,data/bike_day.csv,cnt
```

Rows sharing the same `path` are treated as one imputation batch — this matches the paper's approach of imputing correlated variables from the same source together (e.g. these four Bike Sharing columns) before splitting them into separate series.

**Step 2 — run it:**

```bash
python imbalance_eval.py --config manifest.csv --threshold 0.9 --k 10 --output results.csv
```

**Step 3 — read the output**: one row per manifest entry, same columns as single-file mode.

**Optional: per-dataset overrides.** Add `threshold` and/or `k` columns to override the global default for specific rows (leave blank to fall back to the CLI flag):

```csv
id,name,granularity,path,target,threshold
DS1,Temperature,Daily,data/bike_day.csv,temp,0.9
DS2,Humidity,Daily,data/bike_day.csv,hum,0.95
```

### All CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--target` | *(required in single-file mode)* | Column to evaluate |
| `--config` | *(none)* | Path to a batch manifest CSV (mutually exclusive with a positional CSV path) |
| `--threshold` | `0.9` | Relevance cutoff for "rare" |
| `--k` | `10` | Time delay embedding window size |
| `--no-embed` | off | Skip embedding, score the raw column directly |
| `--xtrm-type` | `both` | Which tail(s) count as extreme: `both`, `high`, or `low` |
| `--coef` | `1.5` | Boxplot whisker coefficient (standard Tukey value) |
| `--date-col` | *(none)* | Date column to sort by and drop before processing |
| `--id` | `DS1` | Label for the output's "ID" column (single-file mode) |
| `--name` | *(target name)* | Label for the output's "Dataset" column (single-file mode) |
| `--granularity` | *(empty)* | Label for the output's "Granularity" column, e.g. `Daily` (single-file mode) |
| `--output` | *(none)* | Also save the results table to this CSV path |

Common mistakes are caught with a clear one-line error instead of a Python traceback: passing a CSV path *and* `--config` together, a `--target`/manifest column that doesn't exist (the error lists the columns that do), a manifest missing a required column, or a `--k` too large for the dataset.

## Validation against the paper

Checked against the real [UCI Bike Sharing dataset](https://doi.org/10.24432/C5W894) (the paper's DS1–DS8 source):

- ✅ **`N` matches exactly.** The paper's Table 1 lists DS1 as having "731 values" — the *raw* series length, before embedding (which would leave 731 − 10 = 721). Table 1's numbers describe the raw series, so use `--no-embed` / `embed=False` when comparing against it.
- ⚠️ **`%Rare` doesn't match at the default coefficient.** At `--coef 1.5` (the standard Tukey value), the daily temperature column has zero boxplot outliers, giving `%Rare=0.0` versus the paper's `9.9%`. The distribution isn't skewed either (skew ≈ −0.05), so this isn't an adjusted-vs-standard-boxplot issue. Smaller `--coef` values land closer to the paper's range (e.g. `coef=0.6` → ~10.3%), but there's no principled way to know the paper's exact original value without their source — `--coef` is exposed so you can tune it once you have more datasets to cross-check against.

<details>
<summary><b>Ruled out: first-order differencing wasn't the missing step</b></summary>

A prior review claimed the paper differenced the data (`Δy_t = y_t − y_{t-1}`) before computing Table 1, and that this reproduces the published numbers exactly. Checked against primary sources, this doesn't hold up:

- The full paper text has no mention of differencing/detrending as a preprocessing step (searched "differenc", "delta", "stationar", "detrend" — only unrelated descriptive language).
- The paper authors' own released R code ([`TSResampStrat_JDSA2017`](https://github.com/nunompmoniz/TSResampStrat_JDSA2017), `Exps.R`, 2,558 lines) contains no `diff(` call anywhere.
- Empirically: differencing the real `temp` column changes `N` from 731 to 730 (differencing always drops a row), but Table 1 states `N=731`. A fix that contradicts one of the two numbers it's explaining isn't a real fix — even though the resulting `%Rare` (~10.0%) happens to land close to 9.9%.

**Don't add a `--diff` flag based on this.** If it comes up again, this note is the first thing to check.
</details>

<details>
<summary><b>Open question: the exact preprocessing behind Table 1's numbers</b></summary>

Still unresolved. The released `Exps.R` loads a pre-built `.Rdata` file with its `knnImputation()` call commented out — meaning real imputation happened upstream, in a script that isn't public, so whether it ran before or after embedding can't be verified. The paper's prose describes imputation before embedding, matching what this tool does, but that hasn't been cross-checked against the actual data.
</details>

## Known limitations

- **No manual relevance control points** — the paper's method is always automatic (boxplot-derived), so `iblr`'s `method="manual"` option isn't wired up. Adding it later is a small, isolated change.
- **No relevance bins or resampling** — this tool only *measures* imbalance; it doesn't implement the paper's actual resampling algorithms.
- **No dataset auto-download** — bring your own CSVs.
- **KNN imputation is O(n²)** in scikit-learn's default mode — fine at the paper's scale (largest dataset ~240k rows), worth knowing for much larger inputs with missing values.

## Running the tests

```bash
pip install -e ".[dev]"
pytest test_imbalance_eval.py -v
```

15 tests: embedding correctness, relevance/imbalance counting (including edge cases like uniform distributions and infinite IR), imputation, both CLI modes end-to-end via subprocess, and the input-validation error paths.

## References

**Paper this tool implements:** Moniz, N., Branco, P., & Torgo, L. (2017). Resampling strategies for imbalanced time series forecasting. *International Journal of Data Science and Analytics*, 3(3), 161–181. https://doi.org/10.1007/s41060-017-0044-3

**Paper authors' original R code** (used to fact-check preprocessing claims above): https://github.com/nunompmoniz/TSResampStrat_JDSA2017

**Relevance function library:** Branco, P. et al. *ImbalancedLearningRegression* [Python package]. https://pypi.org/project/ImbalancedLearningRegression/ — its `phi()`/`phi_ctrl_pts()` port the method from Ribeiro, R. P. (2011). *Utility-based regression* (PhD dissertation), University of Porto.

**Validation dataset:** Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894 (CC BY 4.0)
