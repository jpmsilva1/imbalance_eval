# imbalance_eval

A small CLI/library tool that measures **how imbalanced a time series dataset is**, using the same relevance-function approach as Moniz, Branco & Torgo (2017), *"Resampling Strategies for Imbalanced Time Series Forecasting"*.

It answers one question: *"In this target variable, what fraction of values are 'rare' (unusually extreme), and how bad is the imbalance?"* — the same descriptive statistic the paper reports in its Table 1 (ID, N, Granularity, n_normal, n_rare, IR, %Rare) for each of its 24 datasets.

---

## 1. The idea, in plain terms

1. Take a time series column (e.g. daily temperature, hourly rental counts).
2. Fit a **relevance function** φ(y) that scores every value from 0 (totally normal) to 1 (totally rare/extreme), based on boxplot statistics (quartiles, IQR, whiskers) of that column's own distribution.
3. Pick a **threshold** (default 0.9, matching the paper). Any value with φ(y) ≥ threshold counts as "rare."
4. Report:
   - **N** — total number of observations
   - **n_normal / n_rare** — how many fall on each side of the threshold
   - **IR** (imbalance ratio) — `n_normal / n_rare`
   - **%Rare** — `100 * n_rare / N`

The relevance function itself (φ) is not reimplemented here — it comes straight from the `ImbalancedLearningRegression` (iblr) PyPI package, which is the Python port of the R `UBL` package the paper's authors used.

---

## 2. How the code is organized

Everything lives in one file, `imbalance_eval.py`. It has two layers:

- **Library functions** (top of the file) — usable standalone from Python, e.g. in a notebook.
- **CLI** (bottom of the file) — a thin `argparse` wrapper around the library, for running from the terminal.

### Library functions, in the order data flows through them

| Function | What it does |
|---|---|
| `load_csv(path, date_col=None)` | Reads a CSV. If you pass `date_col`, it parses that column as a date, sorts the rows chronologically by it, then drops it (so it doesn't get treated as a numeric feature downstream). |
| `impute_dataframe(df, n_neighbors=5)` | Fills missing values using k-Nearest-Neighbors imputation (`sklearn.impute.KNNImputer`), run **jointly across all numeric columns** — matching the paper's design of imputing correlated variables together (e.g. temperature/humidity/windspeed from the same weather station) before splitting them into separate series. Non-numeric columns (like a leftover ID or label column) are left untouched. If there are no missing values, it's a no-op. |
| `time_delay_embedding(series, k=10)` | Converts a single 1D time series into a **supervised regression table**: each row becomes `[y(t-k), ..., y(t-1), y(t)]`. This is what "time delay embedding" means — using the last *k* values to predict the next one. The paper uses k=10. Raises a clear error if `k` is too large for the series (would produce zero usable rows). |
| `compute_imbalance(y, rel_thres=0.9, rel_xtrm_type="both", rel_coef=1.5)` | The core calculation. Calls `iblr.phi_ctrl_pts()` then `iblr.phi()` to get a relevance score for every value in `y`, counts how many clear the threshold, and returns a dict: `{N, n_normal, n_rare, IR, %Rare}`. |
| `evaluate_datasets(datasets, k=10, rel_thres=0.9, embed=True, ...)` | Runs the whole pipeline over a **list** of datasets and returns one summary table (a pandas DataFrame) — this is what produces the paper-style Table 1 output. Each entry in `datasets` is a dict: `{"id": ..., "name": ..., "granularity": ..., "series": pd.Series}`. Set `embed=False` to skip step 3 and compute relevance directly on the raw series (useful for quick checks, or for reproducing the paper's own Table 1 numbers, which turn out to describe the *raw* series — see §5 below). |

### CLI functions (wire the library up to the terminal)

| Function | What it does |
|---|---|
| `_get_target_series(df, target, context)` | Looks up a column by name; raises a clear error (listing available columns) if it doesn't exist, instead of a cryptic `KeyError`. |
| `_evaluate_one(...)` | Shared helper: wraps one dataset and calls `evaluate_datasets` — used by both CLI modes below so the wiring logic isn't duplicated. |
| `_run_single(args)` | Handles single-file mode: load → impute → extract target column → evaluate. |
| `_run_batch(args)` | Handles batch mode: reads a CSV manifest, validates it has all required columns, groups rows by source file (so each file is only imputed *once*, jointly across the columns it's used for), then evaluates every row. |
| `main(argv=None)` | Parses arguments, decides single-file vs. batch mode, prints the results table, optionally saves it to CSV. |

### The numpy compatibility shim

`ImbalancedLearningRegression` 0.0.2 (the latest release on PyPI) calls `numpy.quantile(..., interpolation=...)` — a keyword argument that numpy renamed to `method=` and then removed entirely in numpy 2.x. Since this environment has numpy 2.5, that call would crash.

`_iblr_numpy_quantile_compat()` is a small context manager that temporarily patches `numpy.quantile` to translate the old keyword to the new one, **only for the duration of the iblr call** inside `compute_imbalance` — not globally. It restores the original `numpy.quantile` immediately afterward, so importing this module never silently changes numpy's behavior for unrelated code elsewhere in the same process (e.g. other cells in a Jupyter notebook). This can be deleted once iblr ships a numpy-2-compatible release.

---

## 3. Installation

```bash
pip install ImbalancedLearningRegression pandas numpy scikit-learn
```

(`pandas`, `numpy`, and `scikit-learn` come in automatically as transitive dependencies of `ImbalancedLearningRegression`, so in practice `pip install ImbalancedLearningRegression` alone is usually enough.)

To run the test suite, also install pytest:

```bash
pip install pytest
```

---

## 4. Step-by-step: using it as a CLI

### 4a. Single-file mode — evaluate one CSV

Use this when you have one CSV and want to check the imbalance of one column in it.

```bash
python imbalance_eval.py path/to/data.csv --target column_name
```

**Step by step:**

1. `data.csv` is loaded.
2. Missing values (if any) are imputed across all numeric columns.
3. The `column_name` column is extracted as the target series.
4. It's time-delay embedded with the default `k=10` (skip this with `--no-embed`).
5. The relevance function is computed and rare/normal cases are counted at the default threshold `0.9`.
6. A one-row results table is printed.

**Useful flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--target` | *(required)* | Name of the column to evaluate |
| `--threshold` | `0.9` | Relevance cutoff for "rare" (matches the paper) |
| `--k` | `10` | Time delay embedding window size (matches the paper) |
| `--no-embed` | off | Skip embedding, compute relevance on the raw column directly |
| `--xtrm-type` | `both` | Which tail(s) of the distribution count as extreme: `both`, `high`, or `low` |
| `--coef` | `1.5` | Boxplot whisker coefficient (standard Tukey value; the paper says "box plot statistics" without specifying this exactly — see §5) |
| `--date-col` | *(none)* | If your CSV has a date column, pass its name here to sort chronologically and drop it before processing |
| `--id` | `DS1` | Label for the "ID" column in the output |
| `--name` | *(target name)* | Label for the "Dataset" column in the output |
| `--granularity` | *(empty)* | Label for the "Granularity" column, e.g. `Daily`, `Hourly` |
| `--output` | *(none)* | Also save the results table to this CSV path |

**Example** — evaluating the daily temperature column from the UCI Bike Sharing dataset:

```bash
python imbalance_eval.py day.csv --target temp --date-col dteday \
    --id DS1 --name Temperature --granularity Daily \
    --threshold 0.9 --output results.csv
```

### 4b. Batch mode — evaluate many datasets at once (reproduces a Table-1-style summary)

Use this when you have several datasets (or several columns from the same file) and want one combined table, like the paper's Table 1.

**Step 1 — write a manifest CSV.** Required columns: `id`, `name`, `granularity`, `path`, `target`. Optional columns: `threshold`, `k` (per-row overrides — leave blank to use the global CLI flag).

```csv
id,name,granularity,path,target
DS1,Temperature,Daily,data/bike_day.csv,temp
DS2,Humidity,Daily,data/bike_day.csv,hum
DS3,Windspeed,Daily,data/bike_day.csv,windspeed
DS4,Count,Daily,data/bike_day.csv,cnt
```

Rows that share the same `path` are imputed **once, jointly**, before their individual target columns are extracted — this matches how the paper treats correlated variables from the same source (e.g. four Bike Sharing columns share one imputation pass).

**Step 2 — run it:**

```bash
python imbalance_eval.py --config manifest.csv --threshold 0.9 --k 10 --output results.csv
```

**Step 3 — read the output**, a table with columns `ID, Dataset, N, Granularity, n_normal, n_rare, IR, %Rare` — one row per manifest entry.

**Per-row overrides:** add `threshold` and/or `k` columns to the manifest to override the global default for specific datasets:

```csv
id,name,granularity,path,target,threshold
DS1,Temperature,Daily,data/bike_day.csv,temp,0.9
DS2,Humidity,Daily,data/bike_day.csv,hum,0.95
```

### 4c. Error handling

The CLI validates common mistakes and prints a clear one-line error (exit code 2) instead of crashing with a Python traceback:

- Passing both a CSV path *and* `--config` at once
- A `--target` (or manifest `target`) column that doesn't exist — the error lists the columns that *do* exist
- A manifest CSV missing a required column (e.g. `granularity`)
- A `--k` too large for the dataset's length

---

## 5. Step-by-step: using it as a library (e.g. in a notebook)

```python
import pandas as pd
from imbalance_eval import load_csv, impute_dataframe, evaluate_datasets

# 1. Load and impute a raw multivariate CSV (all correlated columns together)
raw = load_csv("day.csv", date_col="dteday")
imputed = impute_dataframe(raw)

# 2. Build a list of datasets -- one entry per target column you want evaluated
datasets = [
    {"id": "DS1", "name": "Temperature", "granularity": "Daily", "series": imputed["temp"]},
    {"id": "DS2", "name": "Humidity",    "granularity": "Daily", "series": imputed["hum"]},
]

# 3. Evaluate -- returns a pandas DataFrame, one row per dataset
table = evaluate_datasets(datasets, k=10, rel_thres=0.9)
print(table)

# 4. Or skip embedding, to match the paper's own descriptive Table 1 numbers
#    (see the note below -- Table 1's N values are the raw series length,
#    not the post-embedding length)
raw_table = evaluate_datasets(datasets, rel_thres=0.9, embed=False)
```

---

## 6. Validation against the paper — what we found

Using the real UCI Bike Sharing dataset (the paper's DS1–DS8 source):

- **N matches exactly.** The paper's Table 1 lists DS1 (daily temperature) as having "731 values" — this is the *raw* series length, not the post-embedding length (which would be 731 − k = 721). This tells us Table 1's descriptive statistics characterize the raw series, **before** time delay embedding — so use `embed=False` when trying to reproduce Table 1 numbers directly.
- **%Rare does not match at the default coefficient.** At `rel_coef=1.5` (the standard Tukey boxplot coefficient, and iblr's default), the daily temperature column has zero values outside the boxplot whiskers, giving `%Rare = 0.0`, versus the paper's reported `9.9%`. The temperature distribution isn't even skewed (skew ≈ −0.05), so this isn't a "standard vs. adjusted/medcouple boxplot" issue — skew-adjustment barely changes anything for a near-symmetric distribution. Sweeping `--coef` shows smaller values recover numbers in the paper's range (e.g. `coef=0.6` gives ~10.3%), but there's no principled way to know the paper's exact original coefficient without their source code, so this isn't hardcoded as a new default — `--coef` is exposed precisely so you can tune it once you have more datasets or the original script to cross-check against.

---

## 7. Running the tests

```bash
pip install pytest
pytest test_imbalance_eval.py -v
```

15 tests cover: time delay embedding correctness, relevance/imbalance counting (including edge cases — uniform distributions with zero rare cases, and the infinite-IR case), imputation (including the non-numeric-column-preservation fix), the full CLI in both modes (via subprocess, checking real stdout/exit codes), and the input-validation error paths.

---

## 8. Known limitations / deliberately out of scope

- **Manual relevance control points** (`iblr`'s `method="manual"` option) aren't wired up — the paper's own method is always automatic (boxplot-derived), so this wasn't needed. `rel_xtrm_type`/`rel_coef` stay as pass-through parameters, so adding manual mode later is a small addition, not a rework.
- **Relevance bins** (grouping temporally-contiguous rare/normal observations, used for the paper's actual *resampling* algorithms) are not implemented — this tool only measures imbalance, it doesn't resample.
- **No dataset auto-download.** You bring your own CSVs; fetching reference datasets (like UCI Bike Sharing) for validation is a manual step, not a CLI feature.
- **KNN imputation is O(n²)** in sklearn's default brute-force mode — fine at the paper's scale (largest dataset ~240k rows), but worth knowing if you point this at something much larger and it happens to have missing values.

---

## 9. References

**Paper this tool implements the relevance-function methodology from:**

Moniz, N., Branco, P., & Torgo, L. (2017). Resampling strategies for imbalanced time series forecasting. *International Journal of Data Science and Analytics*, 3(3), 161–181. https://doi.org/10.1007/s41060-017-0044-3

**Library used for the relevance function (φ) itself:**

Branco, P. (ImbalancedLearningRegression contributors). *ImbalancedLearningRegression* [Python package]. PyPI: https://pypi.org/project/ImbalancedLearningRegression/ — GitHub: https://github.com/paobranco/ImbalancedLearningRegression

The `phi()`/`phi_ctrl_pts()` relevance-function implementation in that package is itself a Python port of the R `UBL` package's method, based on:

Ribeiro, R. P. (2011). *Utility-based regression* (PhD dissertation). Department of Computer Science, Faculty of Sciences, University of Porto.

**Validation dataset used in §6:**

Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894 (CC BY 4.0)
