from pathlib import Path
from statistics import mean, stdev
import csv
import json

import matplotlib.pyplot as plt

import plot_existing_paper as P
import style as S


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIG = ROOT / 'figures'
APP = ROOT / 'appendix'
TYPES = ('choice', 'binary', 'score')
HEADERS = ('Choice ↑', 'Binary ↑', 'NMAE ×100 ↓')


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def summary(values):
    return mean(values), stdev(values) if len(values) > 1 else None


def number(mu, sd=None, *, scale=100, digits=2):
    if sd is not None and 0 < sd * scale < .005 and digits == 2:
        digits = 4
    text = f'{mu * scale:.{digits}f}'
    if sd is not None:
        text += f' ± {sd * scale:.{digits}f}'
    return text


def table(filename, columns, values, *, width=S.COL_W, height=2.0,
          folder=APP, widths=None, font=6.4, highlight=None, title=None):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis('off')
    if title:
        ax.set_title(title, loc='left', pad=5)
    cells = ax.table(cellText=values, colLabels=columns, cellLoc='center',
                     colWidths=widths, bbox=(0, 0, 1, .96))
    cells.auto_set_font_size(False)
    cells.set_fontsize(font)
    for (r, c), cell in cells.get_celld().items():
        cell.set_linewidth(.35)
        cell.set_edgecolor('#D8D8D8')
        cell.set_facecolor('#ECEFF2' if r == 0 else '#F6E8EA' if r - 1 == highlight else 'white')
        cell.get_text().set_ha('left' if c == 0 else 'center')
        if r == 0:
            cell.get_text().set_weight('bold')
    fig.subplots_adjust(left=.02, right=.98, top=.95, bottom=.04)
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f'{filename}.pdf', facecolor='white')
    fig.savefig(folder / f'{filename}.png', dpi=300, facecolor='white')
    plt.close(fig)


def write_csv(name, values):
    with (DATA / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main_tables(baseline):
    cells = {}
    reports = []
    for method, label, group in P.METHODS:
        rows = [r for r in baseline if r['method'] == method]
        seeds = sorted({r['seed'] for r in rows})
        for dataset in P.IDS:
            mu, sd = summary([float(r['primary']) for r in rows if r['dataset'] == dataset])
            cells[method, dataset] = mu, sd
        for kind in TYPES:
            vals = [mean(float(r['primary']) for r in rows if r['seed'] == seed and r['task_type'] == kind)
                    for seed in seeds]
            nll = [mean(float(r['NLL']) for r in rows if r['seed'] == seed and r['task_type'] == kind)
                   for seed in seeds]
            mu, sd = summary(vals)
            nm, ns = summary(nll)
            cells[method, kind] = mu, sd
            reports.append(dict(method=method, task_type=kind, n=len(seeds),
                                primary_mean=mu, primary_sample_SD='' if sd is None else sd,
                                NLL_mean=nm, NLL_sample_SD='' if ns is None else ns))
    write_csv('main_ALL18_macro_primary_NLL.csv', reports)
    labels = ['Method'] + [P.PANEL[d]['header'] for d in P.IDS] + ['ALL Ch.', 'ALL Bin.', 'ALL Sc.']
    cols = P.IDS + list(TYPES)
    for show_sd in (False, True):
        rows = []
        for method, label, group in P.METHODS:
            name = label.replace(r'\method{}', 'PCA-T2L').replace(r'Mean $\Delta W$', 'Mean ΔW')
            vals = []
            for col in cols:
                mu, sd = cells[method, col]
                vals.append(f'{mu * 100:.2f}' + (f'\n±{sd * 100:.2f}' if show_sd and sd is not None else ''))
            rows.append([name] + vals)
        table('table_main_baseline' + ('_SD' if show_sd else ''), labels, rows,
              width=S.TEXT_W, height=3.35 if show_sd else 2.45, folder=FIG,
              widths=[.19] + [.81 / 21] * 21, font=4.3, highlight=12)


def ablation():
    rows = read(DATA / 'component_ablation_mean_SD.csv')
    methods = list(dict.fromkeys(r['method'] for r in rows))
    labels = ['Configuration'] + [f'{cohort}\n{metric}' for cohort in ('ALL18', 'ID12', 'OOD6') for metric in HEADERS]
    for show_sd in (False, True):
        vals = []
        for method in methods:
            values = []
            for cohort in ('ALL18', 'ID12', 'OOD6'):
                for kind in TYPES:
                    r = next(r for r in rows if r['method'] == method and r['cohort'] == cohort and r['task_type'] == kind)
                    mu, sd = float(r['mean']), float(r['sd']) if r['sd'] else None
                    values.append(number(mu, sd if show_sd else None))
            vals.append([method.replace(': ', '\n')] + values)
        table('table_component_ablation' + ('_SD' if show_sd else ''), labels, vals,
              width=S.TEXT_W, height=2.55, folder=FIG,
              widths=[.22] + [.78 / 9] * 9, font=5.3 if show_sd else 6.1, highlight=4)


def nll():
    rows = read(DATA / 'main_ALL18_macro_primary_NLL.csv')
    methods = [
        ('Released bare', 'Backbone'),
        ('Ordinary LoRA (1200, 96k)', 'LoRA 1,200'),
        ('Ordinary LoRA (1x)', 'LoRA 1,591'),
        ('Direct T2L (rank64, 0.93M)', 'Direct 0.93M'),
        ('PCA-T2L (K60, 0.93M)', 'PCA-T2L'),
    ]
    values = []
    for method, label in methods:
        values.append([label] + [number(float(r['NLL_mean']),
                           float(r['NLL_sample_SD']) if r['NLL_sample_SD'] else None,
                           scale=1, digits=3)
                       for kind in TYPES
                       for r in [next(x for x in rows if x['method'] == method and x['task_type'] == kind)]])
    table('native_NLL', ['Method', 'Choice ↓', 'Binary ↓', 'Score ↓'], values,
          width=S.TEXT_W, height=1.62, widths=[.25, .25, .25, .25], font=7.0, highlight=4)


def timings():
    rows = read(DATA / 'training_phase_minutes.csv')
    values = []
    for updates in range(0, 1201, 200):
        times = [float(r['student_minutes']) for r in rows if int(r['R_updates']) == updates]
        assert len(times) == 3
        mu, sd = summary(times)
        values.append([f'{updates} / {1200-updates}', f'{4*updates:,}', f'{80*(1200-updates):,}', number(mu, sd, scale=1)])
    table('PCA_training_time', ['R / CE', 'Teacher pairs', 'CE records', 'Student min'],
          values, width=S.TEXT_W, height=2.0, font=7.0, highlight=5)


def configuration():
    fig = plt.figure(figsize=(S.TEXT_W, 5.6))
    paragraphs = [
        ('Experiment configuration',
         'Fixed released Laya backbone and bare description encoder; 60 teachers; rank64; K60;\n'
         '929,984 student parameters; R1000 + CE200; lambda=0; KD=0.\n'
         'R head/trunk LR: 1e-2 / 3e-4. CE head/trunk LR: 3e-4 / 3e-5.\n'
         'Fresh AdamW per phase; warmup10 then constant LR; weight decay0; clip1.\n'
         'Student final checkpoints use seeds42,43,44; deterministic controls have n1.'),
        ('Evaluation and units',
         'Fixed18 contains30,046 records. ID12 has four views per decision type; OOD6 has two.\n'
         'OOD denotes task-view transfer, not corpus-disjoint transfer.\n'
         'Choice: Accuracy. Binary: (Accuracy + Yes-F1 + BalancedAccuracy)/3. Score: range NMAE.\n'
         'All primary values in plots are multiplied by100, including NMAE. Native NLL is unscaled.\n'
         'Each dataset receives equal weight within type/cohort and seed; SD is across actual seeds.'),
        ('Figure configuration',
         'IEEE column3.50in and text7.16in. Times-compatible serif; 8pt labels and7pt ticks.\n'
         'K curves use the same60-teacher basis; the main two-panel curve is paired change fromK1.\n'
         'The raw K appendix retains separate metric scales. Expanded64/128 banks are separate.\n'
         'R/CE uses1,200 nominal updates; R has4 teacher pairs, CE has80 task records per update.\n'
         'Student phase timing excludes offline teachers, basis fitting, startup and evaluation.\n'
         'Geometry cells average all three main student seeds, with a fixed cosine scale[-1,1].'),
    ]
    y = .95
    for title, text in paragraphs:
        fig.text(.06, y, title, fontsize=10, fontweight='bold', va='top')
        fig.text(.06, y-.045, text, fontsize=8, va='top', linespacing=1.55)
        y -= .29
    fig.savefig(APP / 'figure_configuration.pdf', facecolor='white')
    fig.savefig(APP / 'figure_configuration.png', dpi=300, facecolor='white')
    plt.close(fig)


def main():
    S.apply()
    baseline = read(DATA / 'main_baseline_admitted_seed_dataset.csv')
    main_tables(baseline)
    ablation()
    nll()
    timings()
    configuration()


if __name__ == '__main__':
    main()
