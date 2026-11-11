from pathlib import Path
from statistics import mean, stdev
import csv

import matplotlib.pyplot as plt
import numpy as np

import style as S


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('correct_final', 'wrong_type', 'same_type_wrong_semantics', 'fully_wrong', 'random_gibberish')
LABELS = ('Correct\nfinal', 'Wrong\ntype', 'Wrong\nmeaning', 'Wrong\nboth', 'Random\ntext')
TYPES = ('choice', 'binary', 'score')
SEEDS = (43, 44)


def main():
    S.apply()
    with (ROOT / 'data' / 'description_effects_seed_dataset.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 180 and {r['variant'] for r in rows} == set(VARIANTS)
    assert {int(r['seed']) for r in rows} == set(SEEDS)
    baseline = {(int(r['seed']), r['dataset']): float(r['primary'])
                for r in rows if r['variant'] == 'correct_final'}
    macro_rows = []
    for cohort in ('ALL18', 'ID12', 'OOD6'):
        fig, axes = plt.subplots(3, 1, figsize=(S.COL_W, 4.55), sharex=True,
                                 gridspec_kw={'hspace': .25})
        for ax, kind in zip(axes, TYPES):
            centers, error = [], []
            for variant in VARIANTS:
                subset = [r for r in rows if r['variant'] == variant and r['task_type'] == kind
                          and (cohort == 'ALL18' or r['cohort'] == cohort)]
                per_primary, per_drop, per_nll = [], [], []
                for seed in SEEDS:
                    selected = [r for r in subset if int(r['seed']) == seed]
                    assert len(selected) == (6 if cohort == 'ALL18' else 4 if cohort == 'ID12' else 2)
                    per_primary.append(mean(float(r['primary']) for r in selected))
                    per_drop.append(mean((float(r['primary']) - baseline[seed, r['dataset']])
                                         * (1 if kind == 'score' else -1) for r in selected))
                    per_nll.append(mean(float(r['NLL']) for r in selected))
                centers.append(100 * mean(per_drop))
                error.append(100 * stdev(per_drop))
                macro_rows.append(dict(variant=variant, cohort=cohort, task_type=kind,
                    n=2, seeds='43,44', primary_mean=mean(per_primary), primary_sample_SD=stdev(per_primary),
                    paired_drop_mean=mean(per_drop), paired_drop_sample_SD=stdev(per_drop),
                    NLL_mean=mean(per_nll), NLL_sample_SD=stdev(per_nll)))
            x = np.arange(len(VARIANTS))
            ax.bar(x, centers, color=S.TYPE_COLOR[kind], width=.63)
            ax.errorbar(x, centers, yerr=error, fmt='none', color=S.INK, capsize=2,
                        linewidth=.8)
            ax.axhline(0, color=S.GREY_DARK, linewidth=.6)
            ax.grid(axis='y', color=S.GRID, linewidth=.5)
            ax.set_title(S.TYPE_NAME[kind] + ' (paired n=2)', loc='left')
            ax.set_ylabel('NMAE rise ×100' if kind == 'score' else
                          'Accuracy drop (pp)' if kind == 'choice' else 'Composite drop (pp)')
        axes[-1].set_xticks(range(len(VARIANTS)), LABELS)
        fig.subplots_adjust(left=.20, right=.99, top=.96, bottom=.105)
        name = f'description_effects_{cohort}'
        for suffix in ('pdf', 'png'):
            fig.savefig(ROOT / 'figures' / f'{name}.{suffix}', dpi=300, facecolor='white')
        plt.close(fig)
    with (ROOT / 'data' / 'description_effects_cohort_type_mean_SD.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(macro_rows[0]))
        writer.writeheader()
        writer.writerows(macro_rows)


if __name__ == '__main__':
    main()
