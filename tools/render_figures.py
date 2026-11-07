"""Render README figures from published aggregates; optional matplotlib dependency.

No competition images, individual predictions or model execution are used.
English figure labels keep rendering portable; Korean captions are in the docs.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets'
INK = '#172b4d'
MUTED = '#52647b'
BLUE = '#2563eb'
TEAL = '#087f75'
RED = '#c34436'
BG = '#f8fafc'


def read(name):
    return json.loads((ROOT / 'evidence' / name).read_text(encoding='utf-8'))


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def submission_history():
    record = read('public_scores.json')
    rows = record['entries']
    y = [r['public_rmse'] for r in rows]
    x = list(range(len(rows)))
    gain = 100 * (1 - y[-1] / y[0])
    fig = plt.figure(figsize=(11, 5.6), facecolor=BG)
    fig.text(.075, .915, 'SUBMISSION PROGRESS', color=BLUE, weight='bold', fontsize=12)
    fig.text(.075, .84, f'{y[0]:.4f}  →  {y[-1]:.4f}', color=INK, weight='bold', fontsize=27)
    fig.text(.93, .855, f'−{gain:.1f}%', color=TEAL, weight='bold', fontsize=30, ha='right')
    fig.text(.93, .805, 'reported Public RMSE', color=MUTED, fontsize=12, ha='right')
    ax = fig.add_axes([.085, .23, .845, .5], facecolor=BG)
    ax.plot(x, y, color=BLUE, lw=2.5, zorder=2)
    ax.scatter(x[:-1], y[:-1], color=BLUE, s=55, zorder=3)
    ax.scatter(x[-1], y[-1], color=TEAL, s=105, zorder=3)
    for i, value in enumerate(y):
        ax.annotate(f'{value:.4f}', (i, value), xytext=(0, -24 if i == 2 else 13),
                    textcoords='offset points', ha='center', color=TEAL if i == 5 else INK,
                    weight='bold', fontsize=12)
    ax.set_xticks(x, ['Initial\nensemble', '+ TabPFN', 'Scattering\n486', '+ DINO\nLarge', '+ C-RADIO', 'Large + Radio\nD4'])
    ax.tick_params(axis='x', labelsize=11, length=0, pad=10)
    ax.tick_params(axis='y', labelsize=10, colors=MUTED, length=0)
    ax.set_ylim(11.9, 13.05)
    ax.set_xlim(-.3, 5.3)
    ax.set_yticks([12., 12.4, 12.8])
    ax.grid(axis='y', color='#dce5ef', lw=.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.075, .07, 'Six submitted milestones · lower is better · participant-reported scores', color=MUTED, fontsize=11)
    fig.text(.075, .03, 'Public and OOF use different evaluation samples. This is not a controlled ablation.', color=MUTED, fontsize=10)
    save(fig, 'submission-history.png')


def pipeline():
    model = read('final_model.json')
    w = model['weights']
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')

    def box(x, y, width, height, fill, border):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.008,rounding_size=0.012',
                                   fc=fill, ec=border, lw=1.2))

    def arrow(x1, y1, x2, y2):
        ax.annotate('', (x2, y2), (x1, y1), arrowprops={'arrowstyle':'-|>', 'color':'#8b9db4', 'lw':1.4})

    ax.text(.055, .947, 'FINAL INFERENCE PIPELINE', fontsize=14, color=BLUE, weight='bold')
    ax.text(.055, .905, 'Feature extraction and regression are separate stages', fontsize=12, color=MUTED)
    box(.24, .785, .52, .075, '#eaf1fd', '#b4c7ed')
    ax.text(.5, .823, '256 × 256 grayscale image', ha='center', va='center', fontsize=17, color=INK, weight='bold')
    ax.plot([.5, .5], [.777, .741], color='#8b9db4', lw=1.4)
    ax.plot([.19, .81], [.741, .741], color='#8b9db4', lw=1.4)
    branches = [
        (.06, .19, 'Scattering486', 'Existing ensemble', 'Handcrafted + ResNet\nDINO-small + TabPFN', w['scattering486'], '#edf7f5', TEAL),
        (.37, .5, 'DINOv2-Large', '518 px · frozen', 'D4 feature mean\n→ Ridge', w['large518_d4_ridge'], '#eef3fe', BLUE),
        (.68, .81, 'C-RADIO SO400M', '592 px · frozen', 'D4 feature mean\n→ Ridge', w['cradiov4_d4_ridge'], '#f2effc', '#7956b7'),
    ]
    for left, center, title, subtitle, detail, weight, fill, accent in branches:
        arrow(center, .741, center, .68)
        box(left, .41, .26, .26, fill, '#d6e1ec')
        ax.text(center, .623, title, ha='center', color=INK, fontsize=14, weight='bold')
        ax.text(center, .58, subtitle, ha='center', color=MUTED, fontsize=11)
        ax.text(center, .51, detail, ha='center', va='center', color=INK, fontsize=12, linespacing=1.7)
        ax.text(center, .438, f'{weight:.0%}', ha='center', color=accent, fontsize=18, weight='bold')
        ax.plot([center, center], [.401, .36], color='#8b9db4', lw=1.4)
    ax.plot([.19, .81], [.36, .36], color='#8b9db4', lw=1.4)
    arrow(.5, .36, .5, .31)
    box(.22, .185, .56, .115, INK, INK)
    ax.text(.5, .258, 'Fixed weighted prediction', ha='center', color='white', fontsize=17, weight='bold')
    ax.text(.5, .215, f'Hardness proxy · {model["rows"]:,} test rows', ha='center', color='#dce7f5', fontsize=12)
    ax.text(.5, .108, '8GB execution: one encoder at a time · batch 1 · sequential D4 views', ha='center', fontsize=11, color=MUTED)
    ax.text(.5, .065, 'Checks: feature agreement · chunk hashes · ID order · finite values · CSV round trip', ha='center', fontsize=10.5, color=MUTED)
    save(fig, 'model-pipeline.png')


def fisher_tradeoff():
    runs = read('oof_metrics.json')['experiments']['cradio_fisher_d4']
    fig = plt.figure(figsize=(11, 5.3), facecolor=BG)
    fig.text(.065, .925, 'WHY FISHER WAS NOT ADOPTED', fontsize=14, color=BLUE, weight='bold')
    fig.text(.065, .86, 'Improving hard cases did not improve the whole dataset', fontsize=16, color=INK, weight='bold')
    for col, run in enumerate(runs):
        ax = fig.add_axes([.155 + .475 * col, .255, .335, .46], facecolor=BG)
        diagnosis = run['diagnosis']['fisher_replaced_blend']['groups']
        groups = ['common16', 'remaining484', 'noiseQ4']
        delta = [-diagnosis[g]['gain_vs_current'] for g in groups]
        ax.barh(range(3), delta, color=[TEAL if v < 0 else RED for v in delta], height=.48)
        ax.set_yticks(range(3), ['Common 16', 'Other 484', 'High noise Q4'])
        ax.invert_yaxis()
        ax.axvline(0, color='#91a4bc', lw=1)
        ax.set_xlim(-.71, .36)
        ax.set_xticks([-.6, -.3, 0, .3])
        ax.tick_params(axis='both', length=0, labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        total = -run['metrics']['fisher_replaced_blend']['gain_vs_current']
        ax.set_title(f'Seed {run["seed"]}  |  overall +{total:.4f}', loc='left', color=INK, fontsize=12, weight='bold', pad=15)
        for i, value in enumerate(delta):
            ax.text(value + .02, i, f'{value:+.3f}', va='center', color='white' if value < 0 else RED, fontsize=11, weight='bold')
        ax.set_xlabel('Change in RMSE vs retained model', fontsize=10, color=MUTED, labelpad=10)
    fig.text(.065, .125, '← Lower error', color=TEAL, fontsize=12, weight='bold')
    fig.text(.28, .125, 'Higher error →', color=RED, fontsize=12, weight='bold')
    fig.text(.065, .06, 'Same 500 images and reused folds; diagnostic subgroups are not independent holdouts.', fontsize=10, color=MUTED)
    fig.text(.065, .025, 'High-noise Q4 overlaps the other groups. Full global-feature control comparison: docs/EXPERIMENTS.md.', fontsize=9.5, color=MUTED)
    save(fig, 'fisher-tradeoff.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'text.color':INK})
    submission_history()
    pipeline()
    fisher_tradeoff()


if __name__ == '__main__':
    main()
