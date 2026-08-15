import argparse
import contextlib
import csv
import pathlib
import sys

import numpy as np
import pandas as pd
import ImbalancedLearningRegression as iblr
from sklearn.impute import KNNImputer


@contextlib.contextmanager
def _iblr_numpy_quantile_compat():
    # ponytail: iblr 0.0.2 calls np.quantile(interpolation=...), a numpy<2.0
    # kwarg renamed to method= and removed in numpy 2.x. Shim it only around
    # the iblr call (not globally) so other numpy users in the same process
    # -- e.g. a notebook cell importing this module -- see stock np.quantile.
    # Upgrade path: drop this once iblr ships a numpy-2-compatible release.
    original = np.quantile

    def compat(*args, **kwargs):
        if "interpolation" in kwargs:
            kwargs.setdefault("method", kwargs.pop("interpolation"))
        return original(*args, **kwargs)

    np.quantile = compat
    try:
        yield
    finally:
        np.quantile = original


def impute_dataframe(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns
    if not df[numeric_cols].isna().any().any():
        return df
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df = df.copy()
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


def time_delay_embedding(series: pd.Series, k: int = 10) -> pd.DataFrame:
    cols = {f"lag_{i}": series.shift(i) for i in range(k, 0, -1)}
    cols["target"] = series
    embedded = pd.DataFrame(cols).dropna().reset_index(drop=True)
    if embedded.empty:
        raise ValueError(
            f"time_delay_embedding: k={k} too large for series of length {len(series)} "
            "(embedding produced 0 rows)"
        )
    return embedded


def compute_imbalance(
    y: pd.Series,
    rel_thres: float = 0.9,
    rel_xtrm_type: str = "both",
    rel_coef: float = 1.5,
) -> dict:
    with _iblr_numpy_quantile_compat():
        ctrl_pts = iblr.phi_ctrl_pts(y, method="auto", xtrm_type=rel_xtrm_type, coef=rel_coef)
        phi_values = iblr.phi(y, ctrl_pts)

    n = len(y)
    n_rare = sum(1 for p in phi_values if p >= rel_thres)
    n_normal = n - n_rare
    ir = (n_normal / n_rare) if n_rare > 0 else float("inf")

    return {
        "N": n,
        "n_normal": n_normal,
        "n_rare": n_rare,
        "IR": round(ir, 2),
        "%Rare": round(100 * n_rare / n, 2),
    }


def evaluate_datasets(
    datasets: list,
    k: int = 10,
    rel_thres: float = 0.9,
    embed: bool = True,
    rel_xtrm_type: str = "both",
    rel_coef: float = 1.5,
) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        target = ds["series"]
        if embed:
            target = time_delay_embedding(target, k=k)["target"]

        stats = compute_imbalance(
            target, rel_thres=rel_thres, rel_xtrm_type=rel_xtrm_type, rel_coef=rel_coef
        )

        rows.append({
            "ID": ds["id"],
            "Dataset": ds["name"],
            "N": stats["N"],
            "Granularity": ds.get("granularity", ""),
            "n_normal": stats["n_normal"],
            "n_rare": stats["n_rare"],
            "IR": stats["IR"],
            "%Rare": stats["%Rare"],
        })

    return pd.DataFrame(rows)


def load_csv(path, date_col: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).drop(columns=[date_col]).reset_index(drop=True)
    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate time series imbalance via the iblr relevance function phi."
    )
    parser.add_argument("csv_path", nargs="?", help="CSV file to evaluate (single-file mode)")
    parser.add_argument("--target", help="Target column name (single-file mode, required)")
    parser.add_argument("--id", default="DS1", help="Dataset ID for single-file mode (default: DS1)")
    parser.add_argument("--name", default=None, help="Dataset name (default: target column name)")
    parser.add_argument("--granularity", default="", help="Dataset granularity label")
    parser.add_argument("--date-col", default=None, help="Date column to sort by and drop")
    parser.add_argument("--config", default=None, help="Batch CSV manifest: id,name,granularity,path,target[,threshold,k]")
    parser.add_argument("--threshold", type=float, default=0.9, help="Relevance threshold (default: 0.9)")
    parser.add_argument("--k", type=int, default=10, help="Time delay embedding size (default: 10)")
    parser.add_argument("--no-embed", action="store_true", help="Skip time delay embedding")
    parser.add_argument("--xtrm-type", default="both", choices=["both", "high", "low"])
    parser.add_argument("--coef", type=float, default=1.5, help="Boxplot coefficient (default: 1.5)")
    parser.add_argument("--output", default=None, help="Save results table to this CSV path")
    return parser


def _get_target_series(df: pd.DataFrame, target: str, context: str) -> pd.Series:
    if target not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"target column '{target}' not found in {context}. Available columns: {available}")
    return df[target]


def evaluate_csv_files(
    paths: list,
    target,
    date_col: str = None,
    granularity: str = "",
    k: int = 10,
    rel_thres: float = 0.9,
    embed: bool = True,
    rel_xtrm_type: str = "both",
    rel_coef: float = 1.5,
) -> pd.DataFrame:
    if isinstance(target, dict):
        target_by_path = {str(p): col for p, col in target.items()}
    else:
        target_by_path = {str(p): target for p in paths}

    datasets = []
    for path in paths:
        df = impute_dataframe(load_csv(path, date_col=date_col))
        if str(path) not in target_by_path:
            raise ValueError(f"no target column specified for path {path}; target dict is missing this key")
        column = target_by_path[str(path)]
        series = _get_target_series(df, column, path)
        datasets.append({
            "id": pathlib.Path(path).stem,
            "name": column,
            "granularity": granularity,
            "series": series,
        })

    return evaluate_datasets(
        datasets, k=k, rel_thres=rel_thres, embed=embed,
        rel_xtrm_type=rel_xtrm_type, rel_coef=rel_coef,
    )


REQUIRED_MANIFEST_COLUMNS = {"id", "name", "path", "target"}


def _evaluate_one(id_, name, granularity, series, k, rel_thres, args) -> pd.DataFrame:
    datasets = [{"id": id_, "name": name, "granularity": granularity, "series": series}]
    return evaluate_datasets(
        datasets, k=k, rel_thres=rel_thres, embed=not args.no_embed,
        rel_xtrm_type=args.xtrm_type, rel_coef=args.coef,
    )


def _run_single(args) -> pd.DataFrame:
    df = load_csv(args.csv_path, date_col=args.date_col)
    df = impute_dataframe(df)
    series = _get_target_series(df, args.target, args.csv_path)
    return _evaluate_one(
        args.id, args.name or args.target, args.granularity, series, args.k, args.threshold, args
    )


def _run_batch(args) -> pd.DataFrame:
    with open(args.config, newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest {args.config} is missing required column(s): {', '.join(sorted(missing))}")
        manifest_rows = list(reader)

    by_path = {}
    for row in manifest_rows:
        by_path.setdefault(row["path"], []).append(row)

    results = []
    for path, rows in by_path.items():
        imputed = impute_dataframe(load_csv(path))
        for row in rows:
            threshold = float(row["threshold"]) if row.get("threshold") else args.threshold
            k = int(row["k"]) if row.get("k") else args.k
            series = _get_target_series(imputed, row["target"], f"{path} (manifest row {row['id']})")
            results.append(_evaluate_one(row["id"], row["name"], row.get("granularity", ""), series, k, threshold, args))

    return pd.concat(results, ignore_index=True)


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.config and args.csv_path:
        print("error: provide either a CSV path or --config, not both", file=sys.stderr)
        return 2

    try:
        if args.config:
            result = _run_batch(args)
        elif args.csv_path:
            if not args.target:
                print("error: --target is required in single-file mode", file=sys.stderr)
                return 2
            result = _run_single(args)
        else:
            print("error: provide a CSV path or --config", file=sys.stderr)
            return 2
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(result.to_string(index=False))
    if args.output:
        result.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
