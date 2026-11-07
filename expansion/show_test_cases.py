"""Six held-out AsiaEast examples spanning the SSL model's test PSNR distribution."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from cross_region import Samples, scores
from model import CATFormer

torch.set_num_threads(8)
root=Path('runs/transfer_asiaeast_s0')
out=Path('figures/asiaeast_test_cases'); out.mkdir(parents=True,exist_ok=True)
manifest=json.loads(Path('expansion/prepared/manifest.json').read_text())
records=[r for r in manifest['records'] if r['region']=='asiaEast' and r['split']=='test']
saved=json.loads((root/'ssl_finetune/results.json').read_text())['asiaEast']['per_patch']
ranked=sorted(saved,key=lambda r:r['PSNR'])
picks=[ranked[round(q*(len(ranked)-1))] for q in (.05,.2,.4,.6,.8,.95)]
bykey={tuple(r['key']):r for r in records}
ds=Samples('data',[bykey[tuple(p['key'])] for p in picks])
paths={'Original model':'runs/full_catformer_v2_s0/best.pt',
       'Fine-tuned':str(root/'finetune/best.pt'),
       'Self-supervised + fine-tuned':str(root/'ssl_finetune/best.pt')}
models={}
for name,path in paths.items():
    m=CATFormer(); m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)); m.eval();models[name]=m
rgb=lambda x: np.clip(x[[3,2,1]].transpose(1,2,0)*3.5,0,1)
fig,axes=plt.subplots(6,8,figsize=(22,17))
headers=['Cloudy input 1','Cloudy input 2','Cloudy input 3','Least-cloudy input',*paths,'Low-cloud reference']
info=[]
with torch.no_grad():
    for row in range(len(ds)):
        s=ds[row]; b={k:v[None] for k,v in s.items() if isinstance(v,torch.Tensor)}
        target=s['y'].numpy(); frames=s['s2'].numpy()
        lc=int(s['cloud'].mean((1,2,3)).argmin())
        imgs=[*frames,frames[lc]]; values=[None]*3+[scores(s['s2'][lc][None],s['y'][None])[0][1]]
        item=dict(key=ds.records[row]['key'],selection_percentile=[5,20,40,60,80,95][row],scores={})
        for name,m in models.items():
            pred=m(b['s2'],b['s1'],b['cloud'],b['tpos'])[0]
            met=dict(zip(('MAE','PSNR','SSIM_rgb','SAM'),scores(pred,b['y'])[0]))
            item['scores'][name]=met
            imgs.append(pred[0].numpy());values.append(met['PSNR'])
            Image.fromarray((rgb(pred[0].numpy())*255).round().astype('uint8')).save(out/f'case_{row+1}_{name.split()[0]}.png')
        assert abs(item['scores']['Self-supervised + fine-tuned']['PSNR']-picks[row]['PSNR'])<.02
        imgs.append(target);values.append(None)
        for col,(img,psnr) in enumerate(zip(imgs,values)):
            ax=axes[row,col];ax.imshow(rgb(img));ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            label=''
            if col<3:
                cloud=float((s['cloud'][col]>.4).float().mean())*100
                label=f'{cloud:.0f}% cloud; {s["tpos"][col]:+.1f} months'
            elif psnr is not None:label=f'{psnr:.2f} dB'
            if row==0:label=headers[col]+'\n'+label
            ax.set_title(label,fontsize=9)
        roi,tile,patch=ds.records[row]['key']
        axes[row,0].set_ylabel(f'{roi} / {tile}\nPatch {patch}\nP{item["selection_percentile"]}',fontsize=9)
        info.append(item)
fig.suptitle('Held-out AsiaEast test examples | identical RGB stretch for every image\nSix cases at the 5th, 20th, 40th, 60th, 80th and 95th percentiles of self-supervised-model PSNR',fontsize=13)
fig.tight_layout(rect=(.02,0,1,.96))
fig.savefig(out/'asiaeast_test_comparison.png',dpi=180,facecolor='white')
plt.close(fig)
Image.open(out/'asiaeast_test_comparison.png').convert('RGB').save(out/'asiaeast_test_comparison.jpg',quality=92)
(out/'cases.json').write_text(json.dumps(info,indent=2))
print('FIGURE_DONE',str(out),flush=True)
