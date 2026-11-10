from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_data import rows


OUTPUT = Path(__file__).resolve().parents[1] / "figures"
INK = "#182433"
MUTED = "#526172"
ACCENT = "#153d63"
RULE = "#a5b0bd"
HIGHLIGHT = "#e8f2fb"
PCA = "PCA-T2L (K60, 0.93M)"
METRICS = {"choice": "Choice accuracy ↑", "binary": "Binary composite ↑", "score": "Score NMAE ↓"}

DATASETS = (
    ("scienceqa__choice", "ScienceQA", "ID12", "choice"),
    ("cosmosqa__choice", "CosmosQA", "ID12", "choice"),
    ("social_iqa__choice", "Social IQA", "ID12", "choice"),
    ("sick__choice", "SICK", "ID12", "choice"),
    ("news_headlines_sarcasm__binary", "Headline\nSarcasm", "ID12", "binary"),
    ("liar__binary", "LIAR", "ID12", "binary"),
    ("semeval2018_task3_irony__binary", "SemEval\nIrony", "ID12", "binary"),
    ("measuring_hate_speech__binary", "Measuring\nHate Speech", "ID12", "binary"),
    ("holistic_dialogue__score", "Holistic\nDialogue", "ID12", "score"),
    ("epitome__score", "EPITOME", "ID12", "score"),
    ("cefr_sp__score", "CEFR-SP", "ID12", "score"),
    ("realsumm__score", "RealSumm", "ID12", "score"),
    ("scotus__choice", "SCOTUS", "OOD6", "choice"),
    ("snips__choice", "SNIPS", "OOD6", "choice"),
    ("mrpc__binary", "MRPC", "OOD6", "binary"),
    ("amazon_reviews2023__binary", "Amazon\nReviews ’23", "OOD6", "binary"),
    ("cmu_mosei__score", "CMU-MOSEI", "OOD6", "score"),
    ("amazon_reviews2023__score", "Amazon\nReviews ’23", "OOD6", "score"),
)

METHODS = (
    ("Released bare", "Released bare"),
    ("Bare + description", "Bare + description"),
    ("Mean delta W (60)", "Mean ΔW (60)"),
    ("Arrow routing (60, top 4)", "Arrow routing (60, top 4)"),
    ("Ordinary LoRA (1x)", "Ordinary LoRA (1,591 steps)"),
    ("Ordinary LoRA (1200, 96k)", "Ordinary LoRA (1,200 steps)"),
    ("Direct T2L (Indep-128)", "Direct T2L (Indep-128)"),
    ("Direct T2L (FiLM-512)", "Direct T2L (FiLM-512)"),
    ("Direct T2L (Indep-512)", "Direct T2L (Indep-512)"),
    ("Direct T2L (rank64, 0.50M)", "Direct T2L (r=64, 0.50M)"),
    ("Direct T2L (rank64, 0.93M)", "Direct T2L (r=64, 0.93M)"),
    ("Direct T2L (rank64, 2.00M)", "Direct T2L (r=64, 2.00M)"),
    (PCA, "PCA-T2L (K=60, 0.93M)"),
)

COMPONENTS = (
    ("A", "Bare"),
    ("B", "LoRA · CE 1200"),
    ("C", "PCA-T2L · R 1200"),
    ("D", "PCA-T2L · CE 1200"),
    ("E", "PCA-T2L · R 1000 + CE 200"),
    ("F", "Direct T2L · R 1200"),
    ("G", "Direct T2L · CE 1200"),
    ("H", "Direct T2L · R 1000 + CE 200"),
)


def canvas(width, height):
    figure = plt.figure(figsize=(width, height))
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set(xlim=(0, width), ylim=(height, 0))
    axis.axis("off")
    return figure, axis


def label(axis, x, y, value, *, size=8.4, weight="normal", color=INK, align="center"):
    axis.text(x, y, value, fontsize=size, fontweight=weight, color=color,
              ha=align, va="center", linespacing=1.05)


def cell(axis, x, y, mean, sd, kind, best=False, highlight=False):
    scale, digits = (1, 4) if kind == "score" else (100, 2)
    main = f"{float(mean) * scale:.{digits}f}"
    color = ACCENT if highlight else INK
    weight = "bold" if best else "normal"
    if sd == "":
        label(axis, x, y, main, weight=weight, color=color)
    else:
        label(axis, x, y - .068, main, weight=weight, color=color)
        label(axis, x, y + .094, f"±{float(sd) * scale:.{digits}f}", size=7, color=MUTED)


def save(figure, filename):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / f"{filename}.pdf", facecolor="white")
    figure.savefig(OUTPUT / f"{filename}.png", dpi=320, facecolor="white")
    plt.close(figure)


def main_baseline():
    source = rows("main_dataset_primary_NLL_n3_addendum.csv")
    values = {(row["method"], row["dataset"]): row for row in source}
    assert len(source) == len(values) == len(METHODS) * len(DATASETS)
    assert all(values[PCA, dataset]["n_seed"] == "3" for dataset, _, _, _ in DATASETS)
    width, height = 15.05, 5.67
    figure, axis = canvas(width, height)
    left, start, right = .18, 2.82, width - .18
    step = (right - start) / len(DATASETS)
    method_x, seeds_x = left + .08, 2.52
    top, header_end, row_height = .51, 1.18, .315
    bottom = header_end + len(METHODS) * row_height

    label(axis, method_x, .25, "Main baseline results", size=13, weight="bold", align="left")
    axis.axhspan(top, header_end, xmin=left / width, xmax=right / width, color="#f3f6f9")
    axis.hlines([top, header_end, bottom], left, right, color=INK, lw=[.9, .65, .9])
    label(axis, method_x, .95, "Method", weight="bold", align="left")
    label(axis, seeds_x, .95, "Seeds", weight="bold")
    for cohort, col, count in (("ID · 12 datasets", 0, 12), ("OOD · 6 datasets", 12, 6)):
        label(axis, start + (col + count / 2) * step, .63, cohort,
              size=9.1, weight="bold", color=ACCENT)
    for first, count, kind in (
        (0, 4, "choice"), (4, 4, "binary"), (8, 4, "score"),
        (12, 2, "choice"), (14, 2, "binary"), (16, 2, "score"),
    ):
        label(axis, start + (first + count / 2) * step, .80, METRICS[kind],
              size=7.7, weight="bold")
    for i, (_, display, _, _) in enumerate(DATASETS):
        label(axis, start + (i + .5) * step, 1.015, display, size=7.35)

    for i, (method, display) in enumerate(METHODS):
        row_top = header_end + i * row_height
        y = row_top + row_height / 2
        highlight = method == PCA
        if highlight:
            axis.axhspan(row_top, row_top + row_height, xmin=left / width,
                         xmax=right / width, color=HIGHLIGHT)
        elif i % 2 == 1:
            axis.axhspan(row_top, row_top + row_height, xmin=left / width,
                         xmax=right / width, color="#fafbfd")
        if i in (4, 6, 9, 12):
            axis.hlines(row_top, left, right, color=RULE, lw=.5)
        style = {"weight": "bold", "color": ACCENT} if highlight else {}
        label(axis, method_x, y, display, align="left", **style)
        seed_counts = {values[method, dataset]["n_seed"] for dataset, _, _, _ in DATASETS}
        assert len(seed_counts) == 1
        label(axis, seeds_x, y, seed_counts.pop(), **style)
        for c, (dataset, _, cohort, kind) in enumerate(DATASETS):
            row = values[method, dataset]
            assert row["cohort"] == cohort and row["type"] == kind
            best_value = (min if kind == "score" else max)(
                float(values[name, dataset]["mean_primary"]) for name, _ in METHODS
            )
            cell(axis, start + (c + .5) * step, y, row["mean_primary"],
                 row["sample_SD_primary"], kind, float(row["mean_primary"]) == best_value, highlight)
    for col in (0, 4, 8, 12, 14, 16):
        strong = col in (0, 12)
        axis.vlines(start + col * step, top if strong else .72, bottom,
                    color=RULE, lw=.65 if strong else .30)
    label(axis, method_x, 5.48,
          "Choice and binary in percent. Values are mean ± sample SD across seeds; "
          "single-seed rows show the value alone. Bold marks the best mean per dataset.",
          align="left", color=MUTED, size=7.7)
    save(figure, "table_main_baseline")


def component_ablation():
    original = rows("component_ablation_macro.csv")
    update = {
        (row["cohort"], row["type"]): row
        for row in rows("main_aliases_macro_n3_addendum.csv")
        if row["family"] == "Component ablation"
    }
    values = {
        (row["ID"], row["cohort"], row["type"]): (
            row["primary_mean"], row["primary_sample_sd"], row["n"],
        )
        for row in original
    }
    assert len(values) == 72 and len(update) == 9
    for (cohort, kind), row in update.items():
        assert row["n_seed"] == "3"
        values["E", cohort, kind] = row["mean_primary"], row["sample_SD_primary"], row["n_seed"]
    width, height = 9.2, 4.25
    figure, axis = canvas(width, height)
    left, start, right = .18, 2.78, width - .18
    step = (right - start) / 9
    method_x, seeds_x = left + .08, 2.48
    top, header_end, row_height = .53, 1.04, .367
    bottom = header_end + len(COMPONENTS) * row_height
    groups = (("ALL18", "All · 18 datasets"), ("ID12", "ID · 12 datasets"), ("OOD6", "OOD · 6 datasets"))
    short = {"choice": "Choice ↑", "binary": "Binary ↑", "score": "NMAE ↓"}

    label(axis, method_x, .26, "Component ablation", size=13, weight="bold", align="left")
    axis.axhspan(top, header_end, xmin=left / width, xmax=right / width, color="#f3f6f9")
    axis.hlines([top, header_end, bottom], left, right, color=INK, lw=[.9, .65, .9])
    label(axis, method_x, .86, "Configuration", weight="bold", align="left")
    label(axis, seeds_x, .86, "Seeds", weight="bold")
    for i, (_, display) in enumerate(groups):
        label(axis, start + (3 * i + 1.5) * step, .68, display,
              size=8.9, weight="bold", color=ACCENT)
        for j, kind in enumerate(METRICS):
            label(axis, start + (3 * i + j + .5) * step, .89, short[kind], size=8.1, weight="bold")

    for i, (key, method) in enumerate(COMPONENTS):
        row_top = header_end + i * row_height
        y = row_top + row_height / 2
        highlight = key == "E"
        if highlight or i % 2 == 1:
            axis.axhspan(row_top, row_top + row_height, xmin=left / width,
                         xmax=right / width, color=HIGHLIGHT if highlight else "#fafbfd")
        if key in ("C", "F"):
            axis.hlines(row_top, left, right, color=RULE, lw=.5)
        style = {"weight": "bold", "color": ACCENT} if highlight else {}
        label(axis, method_x, y, f"{key}   {method}", align="left", **style)
        counts = {values[key, cohort, kind][2] for cohort, _ in groups for kind in METRICS}
        assert len(counts) == 1
        label(axis, seeds_x, y, counts.pop(), **style)
        for g, (cohort, _) in enumerate(groups):
            for t, kind in enumerate(METRICS):
                mean, sd, _ = values[key, cohort, kind]
                best = (min if kind == "score" else max)(
                    float(values[arm, cohort, kind][0]) for arm, _ in COMPONENTS
                )
                cell(axis, start + (3 * g + t + .5) * step, y, mean, sd, kind,
                     float(mean) == best, highlight)
    for c in (0, 3, 6):
        axis.vlines(start + c * step, top, bottom, color=RULE, lw=.55)
    label(axis, method_x, 4.13,
          "R = teacher reconstruction. Choice and binary in percent. Mean ± sample SD across seeds.",
          align="left", color=MUTED, size=7.7)
    save(figure, "table_component_ablation")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    main_baseline()
    component_ablation()
