from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator

from figure_data import rows


FIGURES = Path(__file__).resolve().parents[1] / "figures"
DIMENSIONS = (1, 2, 4, 8, 16, 32, 60)
SEEDS = (42, 43, 44)
TYPES = ("choice", "binary", "score")
STYLES = {
    "choice": ("#2566A5", "o"),
    "binary": ("#B15323", "s"),
    "score": ("#087D72", "D"),
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.3,
        "ytick.major.size": 2.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def dimension_summaries():
    cohort = {
        row["dataset_id"]: row["cohort"]
        for row in rows("geometry_n2_task_ID_legend.csv")
    }
    assert len(cohort) == 18
    dataset_values = defaultdict(dict)
    for row in rows("K_dimension_seed_dataset_scalars.csv"):
        dataset = row["dataset_id"]
        key = (cohort[dataset], int(row["K"]), row["task_type"], int(row["seed"]))
        assert dataset not in dataset_values[key]
        dataset_values[key][dataset] = float(row["primary"])

    results = {}
    for group, expected_datasets in (("ID12", 4), ("OOD6", 2)):
        for dimension in DIMENSIONS[:-1]:
            for task_type in TYPES:
                means = []
                for seed in SEEDS:
                    values = dataset_values[group, dimension, task_type, seed]
                    assert len(values) == expected_datasets
                    means.append(mean(values.values()))
                results[group, dimension, task_type] = (mean(means), stdev(means))

    aliases = rows("main_aliases_macro_n3_addendum.csv")
    for row in aliases:
        if row["family"] != "K dimension" or row["cohort"] == "ALL18":
            continue
        assert row["original_row"] == "N60/K60" and int(row["n_seed"]) == 3
        key = (row["cohort"], 60, row["type"])
        assert key not in results
        results[key] = (float(row["mean_primary"]), float(row["sample_SD_primary"]))
    assert len(results) == 2 * len(DIMENSIONS) * len(TYPES)
    return results


def ratio_summaries():
    results = {}
    for row in rows("ratio_macro_primary_NLL_n3_addendum.csv"):
        if row["cohort"] != "ALL18":
            continue
        reconstruction = int(row["R_updates"])
        assert reconstruction + int(row["CE_updates"]) == 1200
        assert int(row["datasets"]) == 6 and int(row["n_seed"]) == 3
        key = (reconstruction, row["type"])
        assert key not in results
        results[key] = (float(row["mean_primary"]), float(row["sample_SD_primary"]))
    assert set(results) == {
        (step, task_type) for step in range(0, 1201, 200) for task_type in TYPES
    }
    return results


def panel_pair(title, x_label):
    figure, (rates, score) = plt.subplots(1, 2, figsize=(3.45, 2.14))
    figure.subplots_adjust(left=0.135, right=0.975, bottom=0.22, top=0.79, wspace=0.49)
    figure.suptitle(title, y=0.973, fontsize=8.2, fontweight="semibold")
    handles = [
        Line2D([], [], color=STYLES[kind][0], marker=STYLES[kind][1], linewidth=1.2, markersize=3.0, label=label)
        for kind, label in (
            ("choice", "Choice accuracy"),
            ("binary", "Binary composite"),
        )
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.525, 0.892),
        ncol=2,
        frameon=False,
        fontsize=6.6,
        handletextpad=0.45,
        columnspacing=1.05,
        borderaxespad=0,
    )
    figure.supxlabel(x_label, y=0.065, fontsize=7.2)
    for axis in (rates, score):
        axis.set_box_aspect(1)
        axis.grid(axis="y", color="#E3E7EB", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.tick_params(pad=2.4, color="#3A4047")
    rates.set_ylabel("Classification (%)", labelpad=2.4)
    score.set_ylabel("Score NMAE", labelpad=2.1)
    return figure, rates, score


def series(axis, xs, values, task_type):
    color, marker = STYLES[task_type]
    factor = 1 if task_type == "score" else 100
    y, sd = zip(*values)
    axis.errorbar(
        xs,
        [factor * value for value in y],
        yerr=[factor * value for value in sd],
        color=color,
        marker=marker,
        linewidth=1.15,
        markersize=2.8,
        markeredgewidth=0.45,
        capsize=1.7,
        capthick=0.65,
        elinewidth=0.75,
        zorder=3,
    )


def save(figure, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.pdf", facecolor="white")
    figure.savefig(FIGURES / f"{stem}.png", facecolor="white", dpi=450)
    plt.close(figure)


def draw_dimension(values, group, label):
    figure, rates, score = panel_pair(f"PCA dimension scan · {label}", "PCA dimension K")
    for task_type in TYPES:
        axis = score if task_type == "score" else rates
        series(axis, DIMENSIONS, [values[group, k, task_type] for k in DIMENSIONS], task_type)
    for axis in (rates, score):
        axis.set_xscale("log", base=2)
        axis.set_xlim(0.82, 72)
        axis.set_xticks(DIMENSIONS, labels=[str(k) for k in DIMENSIONS])
        axis.xaxis.set_minor_locator(NullLocator())
    rates.set_ylim(39, 84)
    rates.set_yticks((40, 50, 60, 70, 80))
    score.set_ylim(0.108, 0.248)
    score.set_yticks((0.12, 0.16, 0.20, 0.24))
    score.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    save(figure, f"K_scan_{label}")


def draw_ratio(values):
    figure, rates, score = panel_pair(
        "Teacher reconstruction and CE · ALL18", "Teacher reconstruction share of 1,200 updates (%)"
    )
    steps = tuple(range(0, 1201, 200))
    shares = [100 * step / 1200 for step in steps]
    for task_type in TYPES:
        axis = score if task_type == "score" else rates
        series(axis, shares, [values[step, task_type] for step in steps], task_type)
    for axis in (rates, score):
        axis.set_xlim(-7, 107)
        axis.xaxis.set_major_locator(FixedLocator(shares))
        axis.set_xticklabels([str(round(value)) for value in shares])
    rates.set_ylim(55, 82)
    rates.set_yticks((55, 60, 65, 70, 75, 80))
    score.set_ylim(0.1175, 0.134)
    score.set_yticks((0.120, 0.124, 0.128, 0.132))
    score.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    save(figure, "reconstruction_CE_ratio")


def main():
    dimensions = dimension_summaries()
    draw_dimension(dimensions, "ID12", "ID")
    draw_dimension(dimensions, "OOD6", "OOD")
    draw_ratio(ratio_summaries())


if __name__ == "__main__":
    main()
