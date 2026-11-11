from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

from pypdf import PdfWriter


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
PAGES = (
    'figures/table_main_baseline.pdf',
    'figures/table_main_baseline_SD.pdf',
    'figures/table_component_ablation.pdf',
    'figures/table_component_ablation_SD.pdf',
    'figures/fig_gain_over_lora.pdf',
    'figures/fig_capacity.pdf',
    'appendix/fig_capacity_raw_type_panels.pdf',
    'figures/fig_allocation.pdf',
    'figures/fig_network_capacity.pdf',
    'appendix/teacher_library_dimension_ALL18.pdf',
    'figures/fig_geometry.pdf',
    'figures/fig_factor_geometry.pdf',
    'figures/description_effects_ALL18.pdf',
    'figures/description_effects_ID12.pdf',
    'figures/description_effects_OOD6.pdf',
    'appendix/native_NLL.pdf',
    'appendix/PCA_training_time.pdf',
    'appendix/figure_configuration.pdf',
)
EXTRA_PAGES = (
    'fig_adapter_strength_ALL18.pdf',
    'fig_injection_components_ALL18.pdf',
    'fig_efficiency_cold.pdf',
    'fig_efficiency_resident_cache.pdf',
    'fig_efficiency_stages.pdf',
    'fig_efficiency_storage.pdf',
)


def main():
    parser = ArgumentParser()
    parser.add_argument('--assemble-only', action='store_true')
    args = parser.parse_args()
    extra = ROOT / 'new_experiments'
    runner = extra / 'plot_new_mechanisms.py'
    if not args.assemble_only:
        for script in ('plot_existing_paper', 'public_tables', 'description_effects'):
            subprocess.run([sys.executable, '-B', str(CODE / f'{script}.py')], check=True)
        if runner.exists():
            subprocess.run([sys.executable, '-B', str(runner), '--data', str(extra / 'data'),
                            '--output', str(extra / 'figures')], check=True)
    writer = PdfWriter()
    for name in PAGES:
        writer.append(ROOT / name, outline_item=Path(name).stem.replace('_', ' '))
    if runner.exists():
        for name in EXTRA_PAGES:
            path = extra / 'figures' / name
            if path.exists():
                writer.append(path, outline_item=path.stem.replace('_', ' '))
    writer.add_metadata({'/Title': 'SchemaJEV Fixed18 Evaluation Figures and Tables'})
    with (ROOT / 'paper_figures.pdf').open('wb') as stream:
        writer.write(stream)


if __name__ == '__main__':
    main()
