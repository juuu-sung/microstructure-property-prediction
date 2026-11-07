"""Explain fixed v2 results, retaining failed gates and reporting sensitivity.

The diagnostic three-way status is post hoc, NOT a replacement validation gate.
Perturbation ranges are sensitivity ranges, not statistical confidence intervals.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if __package__:
    from . import audit_boundary_measurements_v2 as b
else:
    import audit_boundary_measurements_v2 as b


def main():
    plan=b.validate();records=[]
    for subset in ('development','check'):
        original=pd.read_csv(b.OUT/subset/'measurements.csv').set_index('ID')
        stored=pd.read_csv(b.OUT/subset/'stability.csv')
        for r in [r for r in plan['records'] if r['subset']==subset]:
            ident=r['ID'];gray=b.v1.old.read_image(b.ROOT/'open/train'/(ident+'.png'))
            base=original.loc[ident];details=[]
            for name in ('contrast_0.85_plus19','gaussian_noise_sd3_uint8'):
                m,_=b.detect(b.v1.perturb(gray,name,ident))
                expected=stored[(stored.ID==ident)&(stored.variant==name)].iloc[0]
                assert m['status']==expected.variant_status
                details.append({'variant':name,'raw_spacing':m['raw_spacing'],'reported_spacing':m['reported_spacing'],
                                'structure_noise_ratio':m['structure_noise_ratio'],'reasons':m['reasons'],
                                'status':m['status'],'map_f1':float(expected.v2_map_f1),'spacing_change':float(expected.v2_spacing_change)})
            selected=stored[(stored.ID==ident)&stored.variant.isin(['contrast_0.85_plus19','gaussian_noise_sd3_uint8'])]
            statuses=[base.status]+[x['status'] for x in details]
            if len(set(statuses))>1:category='borderline_status_switch'
            elif base.status=='abstain':category='insufficient_observation'
            elif all(x['map_f1']>=.8 and x['spacing_change']<=.15 for x in details):category='stable_under_two_mild_perturbations'
            else:category='unstable_measurement'
            values=[float(base.raw_spacing)]+[x['raw_spacing'] for x in details if x['raw_spacing'] is not None]
            records.append({'ID':ident,'subset':subset,'frozen_status':base.status,'diagnostic_category':category,
                            'raw_spacing':float(base.raw_spacing),'mild_sensitivity_min':min(values),'mild_sensitivity_max':max(values),
                            'base_structure_noise_ratio':float(base.structure_noise_ratio),'perturbations':details})
    b.v1.save(b.OUT/'uncertainty_review.json',{'interpretation':'Post-hoc descriptive classification only; no threshold changes or promotion. Range is over original and two mild perturbations, NOT a confidence interval. Stable boundary measurements may still be semantically wrong.','records':records})
    simple=pd.DataFrame([{k:v for k,v in r.items() if k!='perturbations'} for r in records])
    simple.to_csv(b.OUT/'uncertainty_review.csv',index=False)
    ids=['TRAIN_000041','TRAIN_000380','TRAIN_000319','TRAIN_000159']
    fig,axs=plt.subplots(4,3,figsize=(10,12),layout='constrained')
    for row,ident in enumerate(ids):
        r=next(r for r in records if r['ID']==ident);gray=b.v1.old.read_image(b.ROOT/'open/train'/(ident+'.png'))
        with np.load(b.OUT/r['subset']/'maps.npz',allow_pickle=False) as masks:
            old=np.asarray(b.v1.color(gray,masks[ident+'_v1_line'],(0,230,240)))
            new=np.repeat(gray[...,None],3,axis=2)
            rgb=(255,160,30) if r['diagnostic_category']=='borderline_status_switch' else ((255,60,80) if r['frozen_status']=='abstain' else (50,240,130))
            new[masks[ident+'_boundary']]=rgb
        for col,panel in enumerate([gray,old,new]):
            axs[row,col].imshow(panel,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
            if row==0:axs[row,col].set_title(['Original','Previous dark lines','Mixed boundaries + uncertainty'][col])
        axs[row,0].set_ylabel(ident)
        axs[row,2].set_xlabel({'stable_under_two_mild_perturbations':'Stable under mild perturbations',
                            'borderline_status_switch':'Borderline: status changes under noise',
                            'insufficient_observation':'Insufficient observation: spacing withheld',
                            'unstable_measurement':'Unstable measurement'}[r['diagnostic_category']],fontsize=9)
    fig.suptitle('Boundary proxies: green stability is NOT proof of grain segmentation')
    fig.savefig(b.OUT/'comparison.png',dpi=150);plt.close(fig)
    fresh=[r for r in records if r['subset']=='check']
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for i,r in enumerate(fresh):
        before=r['base_structure_noise_ratio'];after=r['perturbations'][1]['structure_noise_ratio']
        c='#dd8a00' if 'borderline' in r['diagnostic_category'] else '#385570'
        axs[0].plot([i,i],[before,after],color=c,linewidth=2);axs[0].scatter([i,i],[before,after],color=c,marker='o',s=20)
    axs[0].axhline(1.5,color='#c04040',linestyle='--',label='Frozen threshold1.5')
    axs[0].set_xticks(range(10),[r['ID'][-3:] for r in fresh],rotation=45)
    axs[0].set(xlabel='Fresh check train ID suffix',ylabel='Structure/noise proxy before and after noise')
    axs[0].legend(fontsize=8)
    s=pd.read_csv(b.OUT/'check/stability.csv');n=s[s.variant=='gaussian_noise_sd3_uint8']
    axs[1].scatter(100*n.v1_spacing_change,100*n.v2_spacing_change,color='#327eaf')
    axs[1].plot([0,40],[0,40],'--',color='gray');axs[1].set(xlabel='Previous spacing change (%)',ylabel='Mixed-boundary spacing change (%)',xlim=(0,40),ylim=(0,40))
    fig.suptitle('Held measurement settings: improved spacing stability, brittle status boundary')
    fig.savefig(b.OUT/'diagnostics.png',dpi=160);fig.savefig(b.OUT/'diagnostics.pdf');plt.close(fig)
    # Verify all frozen primary outputs remain unchanged after diagnostic calculations.
    for subset in ('development','check'):
        summary=json.loads((b.OUT/subset/'summary.json').read_text(encoding='utf-8'))
        for name,digest in summary['artifacts'].items():assert b.v1.sha(b.OUT/subset/name)==digest
    b.validate();print(simple.groupby(['subset','diagnostic_category']).size().to_string())
    print(json.dumps([r for r in fresh if 'borderline' in r['diagnostic_category']],indent=2))


if __name__=='__main__':main()
