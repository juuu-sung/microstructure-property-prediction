"""Render local train-only audit figures from saved diagnostic tables; no model fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/common_failure_diagnosis_20261011'


def pixels(ident):
    with Image.open(ROOT/'open/train'/f'{ident}.png') as im:
        return np.asarray(im.convert('L'))


def run():
    df=pd.read_csv(OUT/'common16_experts.csv')
    df=df[df.seed==42].sort_values('hardness')
    fig,ax=plt.subplots(figsize=(12,8))
    pos=np.arange(len(df))
    ax.hlines(pos,df.expert_min,df.expert_max,color='#B4C3CE',linewidth=8,label='Range of 20 saved models')
    ax.scatter(df.current_prediction,pos,color='#18749B',s=42,zorder=3,label='Current ensemble')
    ax.scatter(df.hardness,pos,color='#D95635',s=80,marker='D',zorder=4,label='Target')
    ax.set_yticks(pos,df.ID.str.replace('TRAIN_','',regex=False))
    ax.invert_yaxis()
    ax.set(xlabel='Hardness / prediction',ylabel='Training image ID',
           title='All 20 models err in the same direction on every common failure | seed42')
    ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.09),ncol=3,frameon=False)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'expert_ranges.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(12,13))
    for ax,(_,row) in zip(axes.flat,df.sort_values('ID').iterrows()):
        ax.imshow(pixels(row.ID),cmap='gray',vmin=0,vmax=255)
        ax.set_title(f'{row.ID}\ny={row.hardness:.2f} | pred={row.current_prediction:.2f}',fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT/'common16.png',dpi=140,bbox_inches='tight')
    plt.close(fig)
    neighbors=pd.read_csv(OUT/'neighbors_common16.csv')
    labels=pd.read_csv(ROOT/'open/train.csv').set_index('ID').hardness
    ids=['TRAIN_000406','TRAIN_000466','TRAIN_000467','TRAIN_000052']
    reps=['hand285','dino_d4_1152','resnet_texture3104']
    fig,axes=plt.subplots(4,4,figsize=(11,14))
    for i,ident in enumerate(ids):
        choices=[('query',ident)]
        for rep in reps:
            row=neighbors[(neighbors.seed==42)&(neighbors.ID==ident)&(neighbors.representation==rep)].iloc[0]
            choices.append((rep,row.neighbor1))
        for ax,(rep,item) in zip(axes[i],choices):
            ax.imshow(pixels(item),cmap='gray',vmin=0,vmax=255)
            ax.axis('off')
            ax.text(0,-.03,f'{rep}\n{item} | y={labels[item]:.2f}',transform=ax.transAxes,va='top',fontsize=10)
    fig.subplots_adjust(wspace=.05,hspace=.25,bottom=.05,top=.995,left=.01,right=.995)
    fig.savefig(OUT/'neighbor_examples.png',dpi=140,bbox_inches='tight')
    plt.close(fig)
    print('Rendered three local diagnostic figures; saved PNG/PDF expert ranges.')


if __name__=='__main__':run()
