"""Aggregate transfer metrics and project embeddings using one frozen PCA basis."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--root',default='runs/transfer_asiaeast_s0')
a=p.parse_args()
root=Path(a.root)
modes=('zero_shot','finetune','ssl_finetune')
results={m:json.loads((root/m/'results.json').read_text()) for m in modes}
summary={m:{r:{k:v for k,v in d.items() if k!='per_patch'} for r,d in rs.items()} for m,rs in results.items()}
(root/'comparison.json').write_text(json.dumps(summary,indent=2))
lines=['# AsiaWest to AsiaEast transfer: initial seed-0 experiment','',
       'All models start from the same saved AsiaWest seed-0 weights. The target split is fixed before evaluation.',
       'Targets are complete, low-cloud references (at most 5% detected cloud), not verified cloud-free ground truth.',
       'These results use a stricter input-quality and target-quality protocol than the October runs and must be compared within this table.', '',
       '| Model | Region | n | PSNR | SSIM RGB | MAE | SAM |',
       '|---|---|---:|---:|---:|---:|---:|']
for m in modes:
    for r,d in summary[m].items():
        x=d['model']
        lines.append(f"| {m} | {r} | {d['n']} | {x['PSNR']:.3f} | {x['SSIM_rgb']:.4f} | {x['MAE']:.5f} | {x['SAM']:.3f} |")
for r,d in summary['zero_shot'].items():
    x=d['least_cloudy']
    lines.append(f"| least cloudy | {r} | {d['n']} | {x['PSNR']:.3f} | {x['SSIM_rgb']:.4f} | {x['MAE']:.5f} | {x['SAM']:.3f} |")
lines.extend(['','Interpretation: compare East zero-shot with adapted East scores, and inspect West retention for forgetting.',
              'The difference between regional scores reflects both geographic shift and differences in cloud cover and scene difficulty.',
              'One seed is exploratory. Embedding clusters do not by themselves demonstrate useful generalisation.'])
(root/'comparison.md').write_text('\n'.join(lines)+'\n')
ref=np.load(root/'zero_shot'/'embeddings.npz')
source=(ref['region']=='asiaWest') & (ref['split']=='train')
x=ref['features'][source].astype(float)
center=x.mean(0)
_,_,vh=np.linalg.svd(x-center,full_matrices=False)
basis=vh[:2].T
np.savez(root/'embedding_projection.npz',center=center,basis=basis)
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained',sharex=True,sharey=True)
for ax,m in zip(axes,modes):
    z=np.load(root/m/'embeddings.npz')
    xy=(z['features']-center)@basis
    for r,c in [('asiaWest','#2166ac'),('asiaEast','#b2182b')]:
        # Display test features only; fit projection on zero-shot source training features.
        mask=(z['region']==r)&(z['split']=='test')
        ax.scatter(xy[mask,0],xy[mask,1],s=14,alpha=.6,color=c,label=r)
    ax.set_title(m.replace('_',' ')); ax.set_xlabel('Source-trained PCA component 1'); ax.legend()
axes[0].set_ylabel('Source-trained PCA component 2')
fig.suptitle('Encoder embeddings: same PCA basis fitted on original source-training features')
fig.savefig(root/'embedding_comparison.png',dpi=180)
plt.close(fig)
print('SUMMARY_DONE',str(root),flush=True)
