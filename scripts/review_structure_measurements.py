"""Post-audit diagnostics only. Does not change the frozen25 measurement profile.

All-offset transects investigate a synthetic-grid aliasing failure, not a promoted
feature. Same25 diagnostics cannot count as a new independent confirmation.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology
if __package__:
    from . import audit_structure_measurements as a
else:
    import audit_structure_measurements as a


def all_offset_intercepts(mask):
    intervals=[];crossings=0;usable=0
    for axis in (mask,mask.T):
        for row in axis:
            centers=a.runs_centers(row)
            crossings+=len(centers)
            if len(centers)>=2:usable+=1
            intervals.extend(np.diff(centers).tolist())
    return {"spacing":float(np.mean(intervals)) if intervals else None,
            "intervals":len(intervals),"crossings":crossings,"usable_transects":usable}


def main():
    folder=a.OUT;plan=a.validate(folder)
    df=pd.read_csv(folder/"measurements.csv");st=pd.read_csv(folder/"stability.csv")
    with np.load(folder/"maps.npz",allow_pickle=False) as cache:
        records=[]
        for r in plan["records"]:
            ident=r['ID']
            records.append({"ID":ident,"subset":r['subset'],**all_offset_intercepts(cache[ident+'_line'])})
        pd.DataFrame(records).to_csv(folder/'all_offset_diagnostic.csv',index=False)
        fixtures=[]
        for offset in (0,16):
            gray=np.full((256,256),180,np.uint8);gray[offset::32,:]=40;gray[:,offset::32]=40
            measures,maps=a.maps_and_metrics(gray)
            fixtures.append({"offset":offset,"expected_spacing":32.,"known_visible_cells":64 if offset==0 else 81,
                             "fixed_transects":measures['line_spacing'],"all_offset":all_offset_intercepts(maps['line']),
                             "ws1_count":measures['ws1_regions'],"ws2_count":measures['ws2_regions']})
        a.save(folder/'synthetic_grid.json',{"purpose":"unit geometry only, not competition training data",
                "fixtures":fixtures,"finding":"Fixed transects can coincide with parallel boundary bands; all-offset aggregation removes this alias in these two fixtures, not missed-boundary errors."})
        # These three illustrations were selected AFTER review for diagnosed failure types.
        ids=['TRAIN_000205','TRAIN_000078','TRAIN_000103']
        fig,axs=plt.subplots(3,4,figsize=(12,9),layout='constrained')
        for row,ident in enumerate(ids):
            gray=a.old.read_image(a.ROOT/'open/train'/(ident+'.png'))
            panels=[gray,cache[ident+'_normalized'],np.asarray(a.color(gray,cache[ident+'_ws1'],(50,240,120))),
                    np.asarray(a.color(gray,cache[ident+'_line'],(0,230,250)))]
            panels[3]=panels[3].copy();panels[3][cache[ident+'_compact']]=[255,40,60]
            for col,p in enumerate(panels):
                axs[row,col].imshow(p[64:192,64:192],cmap='gray',interpolation='nearest',vmin=0,vmax=1 if col==1 else 255)
                axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
                if row==0:axs[row,col].set_title(['Raw center ROI','NLM + contrast','Old watershed','New lines / compact'][col])
                if col==0:axs[row,col].set_ylabel(ident)
        fig.suptitle('Visual diagnostics only: no hardness, no semantic ground truth')
        fig.savefig(folder/'focus_cases.png',dpi=150);plt.close(fig)
    # Fixed exploratory ablation on THREE development images only: no-NLM vs old-NLM.
    # Both share the same percentile normalization and line detector; no thresholds tuned.
    preprocessing=[]
    for ident in ('TRAIN_000205','TRAIN_000162','TRAIN_000490'):
        gray=a.old.read_image(a.ROOT/'open/train'/(ident+'.png'))
        for method in ('raw_percentile','old_nlm_percentile'):
            if method=='raw_percentile':
                raw=gray.astype(float)/255.;lo,hi=np.quantile(raw,[.02,.98]);norm=np.clip((raw-lo)/max(hi-lo,1e-8),0,1)
            else:norm=a.old.normalize(gray)
            sm=ndi.gaussian_filter(norm,1.,mode='reflect')
            mask=a.old.filter_components(morphology.black_tophat(sm,footprint=morphology.disk(3))>.08,8)>0
            preprocessing.append({'ID':ident,'method':method,'line_area_before_compact_removal':float(mask.mean()),**all_offset_intercepts(mask)})
    a.save(folder/'preprocessing_ablation.json',{'scope':'Three development images, exploratory after primary screen, no regression/no parameter tuning; more detected lines may mean noise false positives','records':preprocessing})
    q=df.groupby('noise_quintile').agg(n=('ID','size'),spacing_available=('line_spacing','count'),
               median_ws1_thin_fraction=('ws1_thin_fraction','median'),median_line_area=('line_area','median'))
    q.to_csv(folder/'quality_summary.csv')
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    axs[0].bar(q.index,q.spacing_available,color='#2980b9');axs[0].set(xticks=q.index,ylim=(0,5.5),xlabel='Noise proxy quintile',ylabel='Images with valid spacing / 5')
    axs[1].plot(q.index,100*q.median_ws1_thin_fraction,'o-',color='#c0392b');axs[1].set(xticks=q.index,xlabel='Noise proxy quintile',ylabel='Median thin watershed regions (%)')
    for col,label in [('dark_open_r4_relative_change','Dark area after opening r4'),('line_spacing_relative_change','Line spacing')]:
        values=st[st.perturbation=='gaussian_noise_sd3_uint8'][col].dropna().to_numpy()
        axs[2].plot(np.sort(values)*100,np.arange(1,len(values)+1)/len(values),label=f'{label} (n={len(values)})')
    axs[2].set(xlabel='Change after added noise (%)',ylabel='Fraction of evaluable images',xlim=(0,100));axs[2].legend(fontsize=7)
    fig.suptitle('25 selected examples: descriptive diagnostics, not performance estimates')
    fig.savefig(folder/'diagnostics.png',dpi=160);fig.savefig(folder/'diagnostics.pdf');plt.close(fig)
    # Correct the supplementary fixture record that was revised after the measurement run.
    # The primary measurements and predeclared screens remain exactly unchanged.
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    before=summary['run_artifacts'].get('synthetic_grid.json')
    summary['supplement_note']='Synthetic fixture expanded to offsets0/16 with correct known cell counts. Primary25 measurements, masks and screens unchanged.'
    summary['previous_synthetic_fixture_sha256']=before
    summary['run_artifacts']['synthetic_grid.json']=a.sha(folder/'synthetic_grid.json')
    a.save(folder/'summary.json',summary)
    a.validate(folder)
    print(json.dumps(fixtures,indent=2));print(q.to_string())


if __name__=='__main__':main()
