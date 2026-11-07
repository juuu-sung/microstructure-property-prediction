"""Read-only verification and plots of the frozen v3 quality audit; no fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
if __package__:
    from . import audit_boundary_quality_v3 as q
else:
    import audit_boundary_quality_v3 as q


def verify():
    q.validate()
    for folder in (q.detector.v1.OUT, q.detector.OUT):
        manifest=json.loads((folder/'completion_manifest.json').read_text(encoding='utf-8'))
        for name,digest in manifest['artifacts'].items():
            if q.detector.v1.sha(folder/name)!=digest:raise ValueError('Old artifact changed: '+name)
        for name,digest in manifest['scripts'].items():
            if q.detector.v1.sha(q.ROOT/name)!=digest:raise ValueError('Old source changed: '+name)
    for subset in ('historical','fresh'):
        folder=q.OUT/subset
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if summary['protocol_sha256']!=q.detector.v1.sha(q.OUT/'protocol.json'):raise ValueError('Protocol changed')
        for name,digest in summary['artifacts'].items():
            if q.detector.v1.sha(folder/name)!=digest:raise ValueError('New artifact changed: '+name)
    print('Both previous audits, v3 outputs, source/image bindings and submission preserved')


def plot():
    verify()
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    rows=pd.read_csv(q.OUT/'fresh/measurements.csv')
    checks=pd.read_csv(q.OUT/'fresh/external_checks.csv')
    names=['불안정','경계 사례','반복 안정']
    codes={'unstable_proxy':0,'borderline_proxy':1,'repeatable_proxy':2}
    colors=['#de6370','#edbc67','#75b49c']
    states=np.array([[codes[r.status]]+[codes[checks[(checks.ID==r.ID)&(checks.external==n)].iloc[0].external_status] for n in q.EXTERNAL] for r in rows.itertuples()])
    fig,ax=plt.subplots(1,3,figsize=(13.5,6.1),gridspec_kw={'width_ratios':[1.6,1,1]})
    ax[0].imshow(states,cmap=ListedColormap(colors),norm=BoundaryNorm([-.5,.5,1.5,2.5],3),aspect='auto')
    for i in range(len(rows)):
        for j in range(4):ax[0].text(j,i,names[states[i,j]],ha='center',va='center',fontsize=9)
    ax[0].set(xticks=range(4),xticklabels=['원본','명암 변화','잡음 A','잡음 B'],yticks=range(10),yticklabels=rows.ID.str[-3:],title='외부 변형별 품질 상태')
    for k,column,band,xlim,title in [(1,'support_min',(.70,.85),(0.45,1),'경계 지도 일치도 최솟값'),(2,'spacing_change_max',(.10,.20),(0,.23),'간격의 최대 상대 변화')]:
        a=ax[k];a.axvspan(*band,color='#fff0cb');a.scatter(rows[column],range(10),c=[colors[codes[s]] for s in rows.status],s=65,edgecolors='#263442',zorder=3)
        a.set(ylim=(9.5,-.5),xlim=xlim,yticks=range(10),yticklabels=[],title=title)
        a.grid(axis='x',alpha=.2)
        for v in band:a.axvline(v,color='#999999',ls=':',lw=1)
        if k==2:a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    fig.suptitle('품질 판정 v3 · 처음 확인한 10장\n반복 안정성은 실제 결정립 경계의 정확도가 아닙니다',fontsize=15,y=.99)
    fig.text(.5,.018,'30회 외부 비교: 동일 26 · 인접 등급 이동 4 · 반복 안정↔불안정 등 큰 이동 0 | 모든 영상의 물리적 의미는 미검증',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.055,1,.90]);fig.savefig(q.OUT/'quality_overview.png',dpi=160);fig.savefig(q.OUT/'quality_overview.pdf');plt.close(fig)
    print('Saved quality_overview.png/pdf from stored outputs')


if __name__=='__main__':plot()
