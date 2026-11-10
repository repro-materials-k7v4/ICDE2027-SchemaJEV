from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from figure_data import rows

OUT = Path(__file__).resolve().parent.parent / "figures"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
TASK_LABELS = {
    "scienceqa__choice": "ScienceQA",
    "cosmosqa__choice": "CosmosQA",
    "social_iqa__choice": "SocialIQA",
    "sick__choice": "SICK",
    "news_headlines_sarcasm__binary": "News Sarc.",
    "liar__binary": "LIAR",
    "semeval2018_task3_irony__binary": "SemEval Irony",
    "measuring_hate_speech__binary": "Hate Speech",
    "holistic_dialogue__score": "Holistic Dial.",
    "epitome__score": "EPITOME",
    "cefr_sp__score": "CEFR-SP",
    "realsumm__score": "RealSumm",
    "scotus__choice": "SCOTUS",
    "snips__choice": "SNIPS",
    "mrpc__binary": "MRPC",
    "amazon_reviews2023__binary": "Amazon Bin.",
    "cmu_mosei__score": "CMU-MOSEI",
    "amazon_reviews2023__score": "Amazon Score",
}
SPACES = (
    ("effective_deltaW", "Effective update $\\Delta W$", "geometry_delta_W"),
    ("factor_A", "Factor $A$", "geometry_factor_A"),
    ("factor_B", "Factor $B$", "geometry_factor_B"),
)
INTERVENTIONS = (
    ("wrong_type", "Wrong type"),
    ("same_type_wrong_semantics", "Wrong semantics\n(same type)"),
    ("fully_wrong", "Fully wrong"),
    ("random_gibberish", "Random gibberish"),
    ("near_1", "Near rewrite 1"),
    ("near_2", "Near rewrite 2"),
    ("near_3", "Near rewrite 3"),
)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=360, facecolor="white")
    plt.close(fig)


def heatmaps():
    legend = sorted(rows("geometry_n2_task_ID_legend.csv"), key=lambda row: int(row["task_id"]))
    pairs = rows("geometry_n2_pairwise_mean_SD.csv")
    datasets = [row["dataset_id"] for row in legend]
    labels = [TASK_LABELS[dataset] for dataset in datasets]
    size = len(datasets)
    split = sum(row["cohort"] == "ID12" for row in legend)
    assert size == 18 and split == 12 and len(pairs) == size * (size - 1) // 2
    edges = np.arange(size + 1) - 0.5
    for space, title, filename in SPACES:
        matrix = np.full((size, size), np.nan)
        np.fill_diagonal(matrix, 1.0)
        for row in pairs:
            i, j = int(row["task_id_i"]) - 1, int(row["task_id_j"]) - 1
            assert datasets[i] == row["dataset_i"] and datasets[j] == row["dataset_j"]
            assert i < j and np.isnan(matrix[i, j])
            matrix[i, j] = matrix[j, i] = float(row[f"{space}_cosine_mean"])
        assert np.isfinite(matrix).all() and (np.abs(matrix) <= 1.0 + 1e-9).all()

        fig = plt.figure(figsize=(3.54, 3.51))
        ax = fig.add_axes([0.215, 0.240, 0.616, 0.621])
        cax = fig.add_axes([0.856, 0.240, 0.026, 0.621])
        tiles = ax.pcolormesh(edges, edges, matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks(range(size), labels, rotation=90)
        ax.set_yticks(range(size), labels)
        ax.tick_params(axis="both", length=0, pad=2.5, labelsize=6)
        ax.axhline(split - 0.5, color="#343A40", linewidth=0.6)
        ax.axvline(split - 0.5, color="#343A40", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
        ax.set_title(title, pad=17, fontweight="semibold")
        for begin, end, group in ((0, split, "ID12"), (split, size, "OOD6")):
            ax.text((begin + end) / (2 * size), 1.024, group, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=6.4, color="#474D55")
        bar = fig.colorbar(tiles, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
        bar.ax.tick_params(length=2, pad=2, labelsize=6)
        bar.outline.set_linewidth(0.5)
        fig.text(0.5, 0.022, "Cosine similarity, mean across two seeds",
                 ha="center", color="#474D55", fontsize=7)
        save(fig, filename)


def prompt_interventions():
    baseline = rows("paired_description_n2_macro.csv")
    gibberish = rows("macro.csv")
    values = {}
    for row in [*baseline, *[item for item in gibberish if item["variant"] == "random_gibberish"]]:
        variant, task = row["variant"], row["task_type"]
        if row["cohort"] == "ALL18" and variant in dict(INTERVENTIONS):
            assert int(row["n"]) == 2 and (variant, task) not in values
            values[variant, task] = float(row["delta_mean"]), float(row["delta_sample_sd"])
    assert len(values) == len(INTERVENTIONS) * 3
    positions = np.arange(len(INTERVENTIONS))

    fig, (classification, scoring) = plt.subplots(
        1, 2, figsize=(5.70, 2.82), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    fig.subplots_adjust(left=0.214, right=0.975, bottom=0.175, top=0.79, wspace=0.28)
    fig.suptitle("Effect of changing the task description", x=0.214, y=0.967,
                 ha="left", fontweight="semibold", fontsize=10)
    fig.text(0.214, 0.887, "ALL18 · paired mean ± sample SD across two seeds",
             ha="left", color="#474D55", fontsize=7.3)

    for axis in (classification, scoring):
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#E0E3E6", linewidth=0.5)
        axis.axvline(0, color="#414852", linewidth=0.75, linestyle="--")
        axis.axhline(3.5, color="#E1E4E8", linewidth=0.7)
        axis.set_ylim(len(INTERVENTIONS) - 0.5, -0.5)
        axis.tick_params(axis="y", length=0, labelsize=7)
        axis.tick_params(axis="x", length=2.5, labelsize=7)
        axis.spines["left"].set_visible(False)

    for task, axis, color, marker, offset, label, scale in (
        ("choice", classification, "#286990", "o", -0.15, "Choice", 100),
        ("binary", classification, "#B75A24", "s", 0.15, "Binary", 100),
        ("score", scoring, "#3C7763", "D", 0, "Score", 1),
    ):
        mean, sd = np.asarray([values[key, task] for key, _ in INTERVENTIONS]).T * scale
        axis.errorbar(mean, positions + offset, xerr=sd, color=color, fmt=marker,
                      markersize=3.9, elinewidth=0.9, capsize=2.0, label=label, zorder=3)

    classification.set_yticks(positions, [label for _, label in INTERVENTIONS])
    scoring.set_yticks(positions, [""] * len(positions))
    classification.set_xlim(-42, 3)
    scoring.set_xlim(-0.015, 0.217)
    classification.set_xticks([-40, -30, -20, -10, 0])
    scoring.set_xticks([0, 0.05, 0.10, 0.15, 0.20], ["0", ".05", ".10", ".15", ".20"])
    classification.set_xlabel("Change in classification (pp)")
    scoring.set_xlabel("Change in NMAE")
    classification.legend(loc="lower left", bbox_to_anchor=(0, 1.012), ncols=2,
                          fontsize=7, frameon=False, borderaxespad=0,
                          handletextpad=0.2, columnspacing=0.8)
    save(fig, "prompt_interventions")


if __name__ == "__main__":
    with mpl.rc_context(STYLE):
        heatmaps()
        prompt_interventions()
