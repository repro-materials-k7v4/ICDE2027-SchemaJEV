from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

from pypdf import PdfWriter


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
PAGES = (
    "figures/table_main_baseline.pdf",
    "figures/table_component_ablation.pdf",
    "figures/K_scan_ID.pdf",
    "figures/K_scan_OOD.pdf",
    "figures/reconstruction_CE_ratio.pdf",
    "figures/geometry_delta_W.pdf",
    "figures/geometry_factor_A.pdf",
    "figures/geometry_factor_B.pdf",
    "figures/prompt_interventions.pdf",
    "figures/paired_dataset_delta_vs_LoRA1200.pdf",
    "appendix/K_dimension_paper_2x2.pdf",
    "appendix/ratio_paper_2x2.pdf",
    "appendix/expanded_basis.pdf",
    "appendix/native_NLL.pdf",
    "appendix/PCA_training_time.pdf",
    "appendix/figure_configuration.pdf",
)


def main():
    parser = ArgumentParser()
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    if not args.assemble_only:
        for script in ("tables", "scans", "geometry", "supplement", "original_appendix", "configuration"):
            subprocess.run([sys.executable, "-B", str(CODE / f"{script}.py")], check=True)
    writer = PdfWriter()
    for name in PAGES:
        writer.append(ROOT / name, outline_item=Path(name).stem.replace("_", " "))
    writer.add_metadata({"/Title": "PCA-T2L Fixed18 Figures and Tables"})
    with (ROOT / "paper_figures.pdf").open("wb") as stream:
        writer.write(stream)


if __name__ == "__main__":
    main()
