from pathlib import Path
from textwrap import wrap

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

SECTIONS = (
    ("Main-table experiment", (
        "PCA-T2L uses a frozen backbone and description encoder, rank-64 adapters, "
        "a fixed 60-teacher basis with K = 60, and 929,984 trainable generator parameters.",
        "The student performs 1,000 teacher-factor matching updates followed by 200 task CE updates. "
        "The matching batch contains four teacher-description pairs. The CE batch contains 80 examples.",
        "Ordinary LoRA uses 1,200 or 1,591 task CE updates. The 1,200-step row uses an independent "
        "seed-42 run and step-1,200 checkpoints from seeds 43 and 44. The 1,591-step row uses seeds 42, 43 and 44.",
        "The fixed controls are bare, bare with description, mean effective update from 60 teachers, "
        "and top-four routing over those teachers. Direct T2L rows distinguish original raw-factor "
        "and parameter-matched canonical-target generators.",
    )),
    ("Learning rates", (
        "For PCA-T2L and the trained Direct T2L rows, matching uses head/trunk rates 0.01 / 0.0003. "
        "Task CE uses 0.0003 / 0.00003. AdamW is initialized for each phase with 10 warmup steps, "
        "zero weight decay and gradient clipping at 1.",
        "Ordinary LoRA uses a peak learning rate of 0.0003 and 80 warmup steps, followed by linear "
        "decay on the 1,591-step schedule.",
    )),
    ("ID and OOD split", (
        "The fixed panel contains 30,046 records. ID12 contains 12 internal held-out datasets with "
        "four datasets per task type. OOD6 contains six task-transfer views with two views per type. "
        "Binary and score views of Amazon Reviews occupy distinct columns.",
        "The panel was selected post hoc. ID denotes internal held-out samples and OOD denotes "
        "task-view transfer. Each dataset keeps its fixed evaluation mask, answer options and score bins.",
    )),
    ("Plot configuration", (
        "Choice is accuracy. Binary averages accuracy, Yes-F1 and balanced accuracy. Score uses "
        "range-normalized MAE. Macro figures weight datasets equally within each task type and seed. "
        "Bars and ± values show sample SD over student seeds. Frozen controls have one result.",
        "The ID and OOD dimension figures use the fixed 60-teacher basis and two square panels. "
        "The ratio figure uses ALL18 and 1,200 nominal updates, with classification rates and NMAE "
        "on separate axes. Original appendix figures retain their original K60 and 1,000/200 n = 2 points.",
        "The heatmaps use pairwise cosine similarity over the same 18 correct descriptions and share "
        "a −1 to 1 scale. Geometry and prompt effects use seeds 43 and 44. The prompt effects subtract "
        "each seed's correct-description result. Random gibberish uses one fixed draw per task.",
        "The LoRA difference plot pairs seeds 42, 43 and 44 at 1,200 total updates. Positive values "
        "favor PCA-T2L. Timing covers the CUDA-synchronized student phases.",
    )),
)


def main():
    fig = plt.figure(figsize=(7.18, 8.0))
    fig.text(0.075, 0.945, "Figure configuration", fontsize=17, weight="bold", va="top")
    y = 0.891
    for title, paragraphs in SECTIONS:
        fig.text(0.075, y, title, fontsize=10, weight="bold", va="top")
        y -= 0.028
        for paragraph in paragraphs:
            lines = wrap(paragraph, width=94, break_long_words=False, break_on_hyphens=False)
            fig.text(0.075, y, "\n".join(lines), fontsize=8.35, va="top", linespacing=1.27)
            y -= 0.019 * len(lines) + 0.007
        y -= 0.012
    assert y >= 0.025, y
    fig.savefig(ROOT / "appendix" / "figure_configuration.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
