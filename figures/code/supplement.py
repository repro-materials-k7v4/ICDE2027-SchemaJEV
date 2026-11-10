from pathlib import Path
from statistics import mean, stdev

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from figure_data import rows


ROOT = Path(__file__).resolve().parents[1]
PCA = "PCA-T2L (K60, 0.93M)"
LORA1200 = "Ordinary LoRA (1200, 96k)"
LORA1591 = "Ordinary LoRA (1x)"
TYPES = ("choice", "binary", "score")
DISPLAY_NAMES = [
    "ScienceQA", "CosmosQA", "Social IQA", "SICK",
    "News Sarcasm", "LIAR", "SemEval Irony", "Measuring Hate Speech",
    "Holistic Dialogue", "EPITOME", "CEFR-SP", "RealSumm",
    "SCOTUS", "SNIPS", "MRPC", "Amazon Reviews",
    "MOSEI", "Amazon Reviews",
]
COLOR = {"ID12": "#336CA8", "OOD6": "#BD7130"}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
})


def save(figure, directory, stem):
    output = ROOT / directory
    output.mkdir(exist_ok=True)
    figure.savefig(output / f"{stem}.pdf", facecolor="white")
    figure.savefig(output / f"{stem}.png", dpi=400, facecolor="white")
    plt.close(figure)


def forest():
    raw = rows("main_seed_dataset_n3_addendum.csv")
    data = {(r["method"], r["dataset"], int(r["seed"])): float(r["primary"])
            for r in raw if r["method"] in (PCA, LORA1200)}
    legend = sorted(rows("geometry_n2_task_ID_legend.csv"), key=lambda r: int(r["task_id"]))
    names = dict(zip([r["dataset_id"] for r in legend], DISPLAY_NAMES))
    fig, axs = plt.subplots(1, 3, figsize=(7.18, 3.22), layout="constrained")
    for ax, task, title in zip(axs, TYPES, ("(a) Choice", "(b) Binary", "(c) Score")):
        tasks = [r for r in legend if r["dataset_id"].endswith(f"__{task}")]
        assert len(tasks) == 6
        ax.axvline(0, color="#525A64", linewidth=0.75)
        ax.axhspan(3.5, 5.5, color="#FDF3E8", zorder=0)
        for y, row in enumerate(tasks):
            ds = row["dataset_id"]
            scale = -1 if task == "score" else 100
            paired = [(data[PCA, ds, s] - data[LORA1200, ds, s]) * scale for s in (42, 43, 44)]
            ax.errorbar(mean(paired), y, xerr=stdev(paired), fmt="o", markersize=3.5,
                        capsize=2, linewidth=0.85, color=COLOR[row["cohort"]], zorder=3)
        ax.set(yticks=range(6), yticklabels=[names[r["dataset_id"]] for r in tasks],
               ylim=(5.5, -0.5), title=title,
               xlabel="LoRA − PCA (NMAE)" if task == "score" else "PCA − LoRA (pp)")
        ax.tick_params(axis="y", length=0, labelsize=7.2)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="x", alpha=0.18)
    handles = [Line2D([], [], color=COLOR[c], marker="o", linestyle="none", markersize=4, label=c)
               for c in ("ID12", "OOD6")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    save(fig, "figures", "paired_dataset_delta_vs_LoRA1200")


def table(title, columns, data, widths, stem, size):
    fig, ax = plt.subplots(figsize=size)
    ax.set_axis_off()
    ax.text(0.035, 0.95, title, weight="bold", size=10, va="top", transform=ax.transAxes)
    cells = ax.table(cellText=data, colLabels=columns, colWidths=widths, cellLoc="center",
                     bbox=(0.035, 0.06, 0.93, 0.75))
    cells.auto_set_font_size(False)
    cells.set_fontsize(8)
    for (i, j), cell in cells.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.6)
        cell.set_facecolor("#DBE5EF" if i == 0 else ("#F1F5F8" if i % 2 else "white"))
        cell.get_text().set_ha("left" if j == 0 else "center")
        if i == 0:
            cell.get_text().set_weight("bold")
    save(fig, "appendix", stem)


def nll():
    data = {(r["method"], r["type"]): r
            for r in rows("main_macro_primary_NLL_n3_addendum.csv") if r["cohort"] == "ALL18"}
    labels = [(LORA1200, "LoRA 1,200"), (LORA1591, "LoRA 1,591"), (PCA, "PCA-T2L")]
    cells = []
    for key, label in labels:
        vals = [data[key, t] for t in TYPES]
        cells.append([label, vals[0]["n_seed"],
                      *[f'{float(v["mean_NLL"]):.3f} ± {float(v["sample_SD_NLL"]):.3f}' for v in vals]])
    table("Native NLL on ALL18", ["Method", "n", "Choice ↓", "Binary ↓", "Score ↓"],
          cells, [0.26, 0.10, 0.213, 0.213, 0.214], "native_NLL", (6.4, 1.72))


def timings():
    costs = rows("ratio_training_costs.csv")
    main = rows("main_training_budget_n3_addendum.csv")
    data = []
    for recon in range(0, 1201, 200):
        ce = 1200 - recon
        if recon == 1000:
            phases = [r for r in main if r["method"] == "PCA_main_projected"]
            duration = [sum(float(r["elapsed_seconds_cuda_synchronized"]) for r in phases
                            if int(r["seed"]) == seed) / 60 for seed in (42, 43, 44)]
        else:
            duration = [float(r["sum_training_phase_seconds"]) / 60
                        for r in costs if int(r["R_steps"]) == recon and int(r["CE_steps"]) == ce]
        assert len(duration) == 3
        data.append([f"{recon} / {ce}", f"{4 * recon:,}", f"{80 * ce:,}",
                     f"{mean(duration):.2f} ± {stdev(duration):.2f}"])
    table("PCA student training by reconstruction / CE steps",
          ["R / CE steps", "Teacher pairs", "CE examples", "Stage time (min)"],
          data, [0.24, 0.23, 0.24, 0.29], "PCA_training_time", (5.65, 2.55))


def expanded():
    source = [r for r in rows("expanded_basis_macro_n3.csv") if r["metric"] != "native_NLL"]
    cells = []
    for n, k in ((64, 60), (64, 64), (128, 60), (128, 128)):
        values = [next(r for r in source if int(r["basis_N"]) == n and int(r["K"]) == k
                       and r["type"] == task) for task in TYPES]
        fmt = []
        for value, task in zip(values, TYPES):
            scale = 1 if task == "score" else 100
            decimals = 4 if task == "score" else 2
            fmt.append(f'{float(value["mean"])*scale:.{decimals}f} ± '
                       f'{float(value["sample_sd"])*scale:.{decimals}f}')
        cells.append([f"{n} / {k}", *fmt])
    table("Expanded teacher bases on ALL18", ["N / K", "Choice (%) ↑", "Binary (%) ↑", "Score NMAE ↓"],
          cells, [0.22, 0.26, 0.26, 0.26], "expanded_basis", (5.65, 1.9))


if __name__ == "__main__":
    forest()
    nll()
    timings()
    expanded()
