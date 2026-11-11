"""Check the committed three-experiment summaries against per-seed scalars."""
import csv
from collections import defaultdict
import math
from pathlib import Path
import statistics

DATA = Path(__file__).resolve().parent / "data"


def read(name):
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    if expected is None:
        assert actual == "", (actual, expected)
    else:
        assert math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def same_mode(left, right):
    if left["experiment"] == "strength":
        return float(left["mode"]) == float(right["mode"])
    return left["mode"] == right["mode"]


def main():
    raw = read("adapter_seed_dataset.csv")
    assert len(raw) == 576
    identities = {(r["experiment"], r["mode"], r["seed"], r["dataset"]) for r in raw}
    assert len(identities) == len(raw)
    views = {r["dataset"]: r for r in read("dataset_views.csv")}
    assert len(views) == 18 and sum(int(r["records"]) for r in views.values()) == 30046
    for row in raw:
        view = views[row["dataset"]]
        assert row["cohort"] == view["cohort"] and row["task_type"] == view["task_type"]
        assert row["records"] == view["records"]
        assert math.isfinite(float(row["primary"])) and math.isfinite(float(row["NLL"]))
    checks = 0
    for summary in read("adapter_cohort_type_mean_SD.csv"):
        selected = [r for r in raw if r["experiment"] == summary["experiment"]
                    and same_mode(r, summary)
                    and r["task_type"] == summary["task_type"]
                    and (summary["cohort"] == "ALL18" or r["cohort"] == summary["cohort"])]
        grouped = defaultdict(list)
        for row in selected:
            grouped[row["seed"]].append(row)
        assert len(grouped) == int(summary["n"])
        for members in grouped.values():
            assert len(members) == int(summary["dataset_count"])
        for key in ("primary", "NLL"):
            values = [statistics.mean(float(r[key]) for r in members) for members in grouped.values()]
            close(summary[key + "_mean"], statistics.mean(values))
            close(summary[key + "_sample_SD"], statistics.stdev(values) if len(values) > 1 else None)
            checks += 2
    for summary in read("adapter_dataset_mean_SD.csv"):
        selected = [r for r in raw if all(r[key] == summary[key] for key in ("experiment", "dataset")) and same_mode(r, summary)]
        assert len(selected) == int(summary["n"])
        for key in ("primary", "NLL"):
            values = [float(r[key]) for r in selected]
            close(summary[key + "_mean"], statistics.mean(values))
            close(summary[key + "_sample_SD"], statistics.stdev(values) if len(values) > 1 else None)
            checks += 2
    timings = read("efficiency_resident_timings.csv")
    assert len(timings) == 18 * 11
    for row in timings:
        assert row["dataset"] in views and 1 <= int(row["query_records"]) <= 4
        assert int(row["max_padded_tokens"]) <= 4096 or int(row["query_records"]) == 1
        assert int(row["warmup"]) == 10 and int(row["repeats"]) == 50
        assert float(row["wall_ms_median"]) > 0
        assert int(row["peak_allocated_bytes"]) >= int(row["allocated_baseline_bytes"])
    cold = read("efficiency_cold_stage_summary.csv")
    assert len(cold) == 8 and all(int(row["n"]) == 3 for row in cold)
    print(f"Verified {len(raw)} per-seed rows, {checks} mean/SD values, and all runtime table scopes.")


if __name__ == "__main__":
    main()
