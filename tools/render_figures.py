"""Optional static figures from public aggregates; requires matplotlib."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/assets'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    scores=json.loads((ROOT/'evidence/public_scores.json').read_text())['entries']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4.4),layout='constrained')
    fig.patch.set_facecolor('#f8fafc');ax.set_facecolor('#f8fafc')
    x=list(range(6));y=[a['public_rmse'] for a in scores]
    ax.plot(x,y,color='#2563eb',lw=2.5,marker='o',markersize=7)
    for i,v in enumerate(y):
        ax.annotate(f'{v:.4f}',(i,v),xytext=(0,12 if i!=2 else -23),textcoords='offset points',ha='center',fontsize=10,fontweight='bold')
    ax.set_xticks(x,['Initial\nensemble','+ TabPFN','Scattering\n486','+ DINO\nLarge','+ C-RADIO','Large + Radio\nD4'])
    ax.set_ylim(11.9,13.03);ax.set_ylabel('Public RMSE (lower is better)');ax.grid(axis='y',alpha=.2)
    ax.set_title('Submission history | 6.7% lower reported RMSE',loc='left',fontweight='bold',pad=24)
    fig.text(.5,-.02,'Participant-reported platform scores. Different evaluation set from OOF; no rank or award claim.',ha='center',fontsize=9,color='#475569')
    fig.savefig(OUT/'submission-history.png',dpi=150,bbox_inches='tight');plt.close(fig)
    runs=json.loads((ROOT/'evidence/oof_metrics.json').read_text())['experiments']['cradio_fisher_d4']
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained');fig.patch.set_facecolor('#f8fafc')
    for ax,r in zip(axes,runs):
        ax.set_facecolor('#f8fafc')
        groups=['common16','remaining484','noiseQ4'];d=r['diagnosis']['fisher_replaced_blend']['groups']
        delta=[-d[g]['gain_vs_current'] for g in groups]
        colors=['#0f766e' if v<0 else '#dc2626' for v in delta]
        ax.barh(range(3),delta,color=colors,height=.55)
        ax.set_yticks(range(3),['Common 16','Remaining 484','High-noise Q4']);ax.invert_yaxis();ax.axvline(0,color='#94a3b8',lw=1)
        ax.set_xlim(-.7,.34);ax.grid(axis='x',alpha=.15);ax.set_title(f'Seed {r["seed"]}',loc='left',fontweight='bold')
        ax.set_xlabel('Fisher RMSE minus current RMSE')
        for i,v in enumerate(delta):
            ax.text(v+.02,i,f'{v:+.3f}',va='center',ha='left',fontsize=10,color='white' if v<0 else 'black')
    fig.suptitle('Why Fisher was not adopted: a subgroup trade-off',fontweight='bold',x=.5)
    fig.text(.5,-.03,'Negative = improvement. Same 500 images and reused splits; not an independent holdout.',ha='center',fontsize=9,color='#475569')
    fig.savefig(OUT/'fisher-tradeoff.png',dpi=150,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
