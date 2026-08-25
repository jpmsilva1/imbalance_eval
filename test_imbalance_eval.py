import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from imbalance_eval import (
    compute_imbalance,
    evaluate_csv_files,
    evaluate_datasets,
    impute_dataframe,
    load_csv,
    time_delay_embedding,
)

REPO_ROOT = pathlib.Path(__file__).parent.resolve()


def test_time_delay_embedding_known_lags():
    series = pd.Series([10, 20, 30, 40, 50])
    embedded = time_delay_embedding(series, k=2)

    assert list(embedded.columns) == ["lag_2", "lag_1", "target"]
    assert embedded["lag_2"].tolist() == [10, 20, 30]
    assert embedded["lag_1"].tolist() == [20, 30, 40]
    assert embedded["target"].tolist() == [30, 40, 50]
    assert len(embedded) == len(series) - 2


def test_time_delay_embedding_raises_when_k_too_large():
    series = pd.Series([1, 2, 3, 4, 5])
    try:
        time_delay_embedding(series, k=10)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "k=10" in str(e)
        assert "5" in str(e)


def test_compute_imbalance_uniform_distribution_has_no_rare_cases():
    # A uniform spread has no values beyond the Tukey whiskers, so iblr's
    # auto boxplot control points all carry relevance 0 -> phi is 0 everywhere.
    y = pd.Series(np.linspace(0, 1, 100))
    stats = compute_imbalance(y, rel_thres=0.9)

    assert stats["N"] == 100
    assert stats["n_rare"] == 0
    assert stats["n_normal"] == 100
    assert stats["IR"] == 0
    assert stats["%Rare"] == 0.0


def test_compute_imbalance_detects_outliers_as_rare():
    # 950 values with a small realistic spread plus 50 extreme outliers far
    # beyond the IQR whiskers: outliers must be flagged rare, majority must
    # not. (A literally zero-variance cluster degenerates the boxplot IQR to
    # 0, collapsing iblr's control points onto a single x -- unrealistic for
    # real time series, so the spread here keeps IQR non-zero.)
    y = pd.Series(list(range(45, 55)) * 95 + [100000] * 50)
    stats = compute_imbalance(y, rel_thres=0.9)

    assert stats["N"] == 1000
    assert 0 < stats["n_rare"] < stats["N"]
    assert stats["n_normal"] == stats["N"] - stats["n_rare"]
    assert stats["IR"] == round(stats["n_rare"] / stats["n_normal"], 2)
    assert stats["%Rare"] == round(100 * stats["n_rare"] / stats["N"], 2)


def test_compute_imbalance_accepts_arrays_and_lists():
    # iblr's phi() calls y.reset_index(), so raw numpy arrays and lists would
    # otherwise fail with a confusing AttributeError from inside the dependency.
    values = list(range(45, 55)) * 95 + [100000] * 50
    expected = compute_imbalance(pd.Series(values))

    assert compute_imbalance(np.array(values)) == expected
    assert compute_imbalance(values) == expected


def test_compute_imbalance_rejects_degenerate_input():
    # pd.Series(5.0) is a valid length-1 Series, so without a guard a scalar
    # would report {N: 1, n_rare: 0, IR: 0} rather than failing.
    for bad in (5.0, [], [42.0]):
        try:
            compute_imbalance(bad)
            assert False, f"expected ValueError for {bad!r}"
        except ValueError as e:
            assert "at least 2 values" in str(e)


# First 120 values of diff(day.csv$temp) from the UCI Bike Sharing dataset --
# i.e. DS1 from Moniz, Branco & Torgo (2017) before it's cut down for this
# fixture. Independent reference count (n_rare=9) was computed in R via:
#   ph <- UBL::phi.control(y, method="extremes", extr.type="both", coef=1.5)
#   sum(UBL::phi(y, ph) >= 0.9)
# using UBL (the CRAN successor to the `uba` package the paper's own code
# calls) -- a different implementation of the same relevance function, so
# this isn't circular. See README's "Validation against the paper" section:
# the paper differences its series before scoring them; this library doesn't
# difference for you, so the fixture here is pre-differenced.
_DS1_DIFF_TEMP_FIXTURE = [
    0.019311, -0.167114, 0.003636, 0.026957, -0.022609, -0.007826, -0.031522,
    -0.026667, 0.012500, 0.018258, 0.003636, -0.007727, -0.004130, 0.072463,
    -0.001666, -0.055834, 0.040834, 0.075507, -0.030507, -0.084167, -0.118370,
    0.037391, 0.000870, 0.126087, -0.005978, -0.022500, 0.008478, -0.006956,
    0.020000, -0.035689, 0.011341, 0.067826, -0.073043, 0.024347, 0.022029,
    0.052500, -0.014166, -0.050834, -0.086050, 0.009565, 0.044743, 0.033409,
    0.094022, 0.098478, -0.148913, 0.052174, 0.117572, 0.085834, -0.122500,
    -0.113950, 0.018116, -0.121111, 0.039517, 0.073913, 0.068696, -0.081848,
    0.060978, 0.063795, -0.140606, 0.068333, -0.136667, 0.063334, 0.122500,
    -0.007645, -0.114783, 0.030761, 0.003333, 0.093258, -0.072569, 0.012645,
    0.055181, -0.059131, -0.007826, 0.047826, 0.049783, 0.125000, -0.067500,
    -0.140000, 0.097935, 0.011232, -0.094710, -0.061957, -0.020833, 0.001666,
    -0.012790, 0.011305, 0.038152, -0.002500, -0.031667, 0.031667, 0.015000,
    0.063333, 0.195000, -0.159166, -0.023334, 0.046667, -0.101667, 0.006667,
    0.084167, 0.168985, -0.093152, -0.090000, 0.055000, -0.020833, -0.015834,
    0.025834, 0.055833, -0.006667, 0.089167, -0.135833, -0.122500, 0.123333,
    0.121667, 0.025000, 0.025000, -0.011667, -0.002500, -0.107500, -0.037500,
    -0.020833,
]


def test_compute_imbalance_matches_reference_r_implementation():
    stats = compute_imbalance(_DS1_DIFF_TEMP_FIXTURE, rel_thres=0.9)

    assert stats["N"] == 120
    assert stats["n_rare"] == 9  # R UBL::phi() reference count, see fixture comment above


def test_compute_imbalance_warns_when_no_boxplot_outliers():
    # Same mechanism as raw (non-differenced) daily temp/cnt in the paper's own
    # Table 1: a uniform series has no boxplot outliers, so phi_ctrl_pts emits
    # (min,0),(median,0),(max,0) and phi is 0 everywhere -- n_rare=0 is a
    # property of the data, not a bug, but it's surprising enough to flag.
    with pytest.warns(UserWarning, match="no boxplot outliers"):
        stats = compute_imbalance(list(range(100)))

    assert stats["n_rare"] == 0
    assert stats["IR"] == 0


def test_impute_dataframe_fills_nan_and_preserves_present_values():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, np.nan],
    })

    imputed = impute_dataframe(df, n_neighbors=4)

    assert not imputed.isna().any().any()
    assert imputed["a"].tolist() == [1, 2, 3, 4, 5]
    assert imputed["b"].tolist()[:4] == [10, 20, 30, 40]
    # all 4 other rows used as neighbors -> plain mean of their "b" values
    assert imputed["b"].iloc[4] == 25.0


def test_impute_dataframe_short_circuits_when_no_nans():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    imputed = impute_dataframe(df)
    pd.testing.assert_frame_equal(imputed, df)


def test_impute_dataframe_preserves_non_numeric_columns_untouched():
    # A date/label column alongside numeric data with a gap must not crash
    # KNNImputer (which requires all-numeric input) and must survive as-is.
    df = pd.DataFrame({
        "date": ["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04", "2011-01-05"],
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, np.nan],
    })

    imputed = impute_dataframe(df, n_neighbors=4)

    assert imputed["date"].tolist() == df["date"].tolist()
    assert not imputed["b"].isna().any()
    assert imputed["b"].iloc[4] == 25.0


def test_evaluate_datasets_embeds_and_matches_compute_imbalance():
    series = pd.Series(list(range(45, 55)) * 5 + [100000] * 5)  # 55 values
    datasets = [{"id": "DS1", "name": "Toy", "granularity": "Daily", "series": series}]

    result = evaluate_datasets(datasets, k=3, rel_thres=0.9)

    assert list(result.columns) == [
        "ID", "Dataset", "N", "Granularity", "n_normal", "n_rare", "IR", "%Rare",
    ]
    assert len(result) == 1

    embedded_target = time_delay_embedding(series, k=3)["target"]
    expected = compute_imbalance(embedded_target, rel_thres=0.9)

    row = result.iloc[0]
    assert row["ID"] == "DS1"
    assert row["Dataset"] == "Toy"
    assert row["Granularity"] == "Daily"
    assert row["N"] == expected["N"]
    assert row["n_rare"] == expected["n_rare"]
    assert row["n_normal"] == expected["n_normal"]
    assert row["IR"] == expected["IR"]
    assert row["%Rare"] == expected["%Rare"]


def test_evaluate_datasets_defaults_granularity_when_omitted():
    series = pd.Series(list(range(45, 55)) * 5 + [100000] * 5)
    datasets = [{"id": "DS1", "name": "Toy", "series": series}]  # no "granularity" key

    result = evaluate_datasets(datasets, k=3, rel_thres=0.9)

    assert result.iloc[0]["Granularity"] == ""


def test_evaluate_datasets_can_skip_embedding():
    series = pd.Series(list(range(45, 55)) * 5 + [100000] * 5)
    datasets = [{"id": "DS1", "name": "Toy", "granularity": "Daily", "series": series}]

    result = evaluate_datasets(datasets, rel_thres=0.9, embed=False)

    expected = compute_imbalance(series, rel_thres=0.9)
    assert result.iloc[0]["N"] == expected["N"] == len(series)


def test_evaluate_datasets_diff_scores_first_order_differences():
    series = pd.Series(list(range(45, 55)) * 5 + [100000] * 5)
    datasets = [{"id": "DS1", "name": "Toy", "granularity": "Daily", "series": series}]

    result = evaluate_datasets(datasets, rel_thres=0.9, embed=False, diff=True)

    expected = compute_imbalance(series.diff().dropna(), rel_thres=0.9)
    assert result.iloc[0]["N"] == expected["N"] == len(series) - 1
    assert result.iloc[0]["n_rare"] == expected["n_rare"]


def test_load_csv_round_trips_and_sorts_by_date(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "date,temp,hum\n"
        "2011-01-03,12,50\n"
        "2011-01-01,10,40\n"
        "2011-01-02,11,45\n"
    )

    df = load_csv(csv_path, date_col="date")

    assert list(df.columns) == ["temp", "hum"]
    assert df["temp"].tolist() == [10, 11, 12]
    assert df["hum"].tolist() == [40, 45, 50]


def _make_toy_data_csv(tmp_path):
    n_normal_block = 30
    temp = (list(range(45, 55)) * n_normal_block)[: n_normal_block * 10]
    hum = temp  # same distribution shape, reused as a second target column
    outliers_temp = [100000] * 15
    df = pd.DataFrame({
        "temp": temp + outliers_temp,
        "hum": hum + outliers_temp,
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_evaluate_csv_files_shared_target(tmp_path):
    data_path = _make_toy_data_csv(tmp_path)

    result = evaluate_csv_files([str(data_path)], target="temp", k=2, rel_thres=0.9)

    assert result.iloc[0]["ID"] == data_path.stem
    assert result.iloc[0]["Dataset"] == "temp"
    assert result.iloc[0]["Granularity"] == ""
    assert result.iloc[0]["n_rare"] > 0

    raw = load_csv(data_path)
    imputed = impute_dataframe(raw)
    expected = evaluate_datasets(
        [{"id": "x", "name": "x", "series": imputed["temp"]}], k=2, rel_thres=0.9
    ).iloc[0]
    assert result.iloc[0]["N"] == expected["N"]
    assert result.iloc[0]["n_rare"] == expected["n_rare"]


def test_evaluate_csv_files_per_path_target_override(tmp_path):
    data_path = _make_toy_data_csv(tmp_path)

    result = evaluate_csv_files([str(data_path)], target={str(data_path): "hum"}, k=2, rel_thres=0.9)

    assert result.iloc[0]["Dataset"] == "hum"


def test_evaluate_csv_files_passes_granularity_through(tmp_path):
    data_path = _make_toy_data_csv(tmp_path)

    result = evaluate_csv_files([str(data_path)], target="temp", granularity="Daily", k=2, rel_thres=0.9)

    assert result.iloc[0]["Granularity"] == "Daily"


def test_evaluate_csv_files_missing_target_dict_entry_gives_clear_error(tmp_path):
    data_path = _make_toy_data_csv(tmp_path)

    try:
        evaluate_csv_files([str(data_path)], target={})
        assert False, "expected ValueError"
    except ValueError as e:
        assert str(data_path) in str(e)


def test_evaluate_csv_files_unknown_target_column_gives_clear_error(tmp_path):
    data_path = _make_toy_data_csv(tmp_path)

    try:
        evaluate_csv_files([str(data_path)], target="nope")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "nope" in str(e)
        assert "temp" in str(e)


def test_cli_single_file_mode(tmp_path):
    data_path = _make_toy_data_csv(tmp_path)
    output_path = tmp_path / "results.csv"

    proc = subprocess.run(
        [sys.executable, "imbalance_eval.py", str(data_path),
         "--target", "temp", "--k", "2", "--threshold", "0.9",
         "--output", str(output_path)],
        cwd=REPO_ROOT,
        capture_output=True, text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "DS1" in proc.stdout
    assert output_path.exists()

    result = pd.read_csv(output_path)
    assert list(result.columns) == [
        "ID", "Dataset", "N", "Granularity", "n_normal", "n_rare", "IR", "%Rare",
    ]
    assert result.iloc[0]["ID"] == "DS1"
    assert result.iloc[0]["n_rare"] > 0


def test_cli_single_file_mode_with_diff_flag(tmp_path):
    data_path = _make_toy_data_csv(tmp_path)

    no_diff = subprocess.run(
        [sys.executable, "imbalance_eval.py", str(data_path),
         "--target", "temp", "--no-embed", "--threshold", "0.9"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    with_diff = subprocess.run(
        [sys.executable, "imbalance_eval.py", str(data_path),
         "--target", "temp", "--no-embed", "--diff", "--threshold", "0.9"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )

    assert no_diff.returncode == 0, no_diff.stderr
    assert with_diff.returncode == 0, with_diff.stderr
    assert no_diff.stdout != with_diff.stdout


def test_cli_batch_mode(tmp_path):
    data_path = _make_toy_data_csv(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "id,name,granularity,path,target\n"
        f"DS1,Temperature,Daily,{data_path},temp\n"
        f"DS2,Humidity,Daily,{data_path},hum\n"
    )
    output_path = tmp_path / "results.csv"

    proc = subprocess.run(
        [sys.executable, "imbalance_eval.py", "--config", str(manifest_path),
         "--k", "2", "--threshold", "0.9", "--output", str(output_path)],
        cwd=REPO_ROOT,
        capture_output=True, text=True,
    )

    assert proc.returncode == 0, proc.stderr
    result = pd.read_csv(output_path)
    assert result["ID"].tolist() == ["DS1", "DS2"]
    assert (result["n_rare"] > 0).all()


def test_cli_rejects_csv_path_and_config_together(tmp_path):
    data_path = _make_toy_data_csv(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(f"id,name,granularity,path,target\nDS1,Temp,Daily,{data_path},temp\n")

    proc = subprocess.run(
        [sys.executable, "imbalance_eval.py", str(data_path), "--config", str(manifest_path)],
        cwd=REPO_ROOT,
        capture_output=True, text=True,
    )

    assert proc.returncode == 2
    assert "not both" in proc.stderr.lower()


def test_cli_single_file_unknown_target_column_gives_clear_error(tmp_path):
    data_path = _make_toy_data_csv(tmp_path)

    proc = subprocess.run(
        [sys.executable, "imbalance_eval.py", str(data_path), "--target", "nope"],
        cwd=REPO_ROOT,
        capture_output=True, text=True,
    )

    assert proc.returncode == 2
    assert "nope" in proc.stderr
    assert "temp" in proc.stderr  # lists available columns


def test_cli_batch_mode_without_granularity_column(tmp_path):
    data_path = _make_toy_data_csv(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "id,name,path,target\n"
        f"DS1,Temperature,{data_path},temp\n"
    )
    output_path = tmp_path / "results.csv"

    proc = subprocess.run(
        [sys.executable, "imbalance_eval.py", "--config", str(manifest_path),
         "--k", "2", "--threshold", "0.9", "--output", str(output_path)],
        cwd=REPO_ROOT,
        capture_output=True, text=True,
    )

    assert proc.returncode == 0, proc.stderr
    result = pd.read_csv(output_path)
    assert result.iloc[0]["Granularity"] == "" or pd.isna(result.iloc[0]["Granularity"])


def test_cli_batch_missing_manifest_column_gives_clear_error(tmp_path):
    data_path = _make_toy_data_csv(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    # missing the required "target" column
    manifest_path.write_text(f"id,name,granularity,path\nDS1,Temp,Daily,{data_path}\n")

    proc = subprocess.run(
        [sys.executable, "imbalance_eval.py", "--config", str(manifest_path)],
        cwd=REPO_ROOT,
        capture_output=True, text=True,
    )

    assert proc.returncode == 2
    assert "target" in proc.stderr
