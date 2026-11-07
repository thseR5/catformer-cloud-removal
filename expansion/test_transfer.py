"""Small CPU smoke test; no scientific results are inferred from synthetic data."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from cross_region import ssl_example, scores, train_epochs
from model import CATFormer

torch.set_num_threads(2)
torch.manual_seed(23)
B,T,H,W=2,3,32,32
b=dict(s2=.1+.2*torch.rand(B,T,13,H,W), s1=torch.rand(B,T,2,H,W),
       cloud=torch.zeros(B,T,1,H,W), tpos=torch.tensor([[-1.,0.,1.]]*B),
       y=torch.full((B,13,H,W),99.), index=torch.arange(B))
old=b['s2'].clone()
a,target,mask=ssl_example(b)
assert torch.equal(b['s2'],old), 'SSL mutated original observations'
assert target.max()<1, 'SSL read supervised target'
assert torch.all(a['s2'][mask[:,None].expand_as(a['s2']).bool()]==0), 'Masked data leaked'
assert torch.all(a['cloud'][mask[:,None].expand_as(a['cloud']).bool()]==1)
assert torch.equal(a['tpos'][:,0],torch.zeros(B)), 'Target timestamp was not recentered'
assert max(scores(target,target)[0][0:1])==0
model=CATFormer(widths=(8,16,32,64),depth=1,heads=4)
# Use random 50% masking that has enough blocks for a meaningful optimization step.
H=W=64
samples=[]
for i in range(4):
    samples.append(dict(s2=.1+.2*torch.rand(T,13,H,W), s1=torch.rand(T,2,H,W),
                        cloud=torch.zeros(T,1,H,W), tpos=torch.tensor([-1.,0.,1.]),
                        y=.1+.2*torch.rand(13,H,W), index=i))
dl=DataLoader(samples,batch_size=2)
args=SimpleNamespace(lr=3e-5,ssl_epochs=2,epochs=2)
with tempfile.TemporaryDirectory() as d:
    out=Path(d)
    train_epochs(model,dl,dl,args,out,ssl=True)
    train_epochs(model,dl,dl,args,out)
    assert (out/'ssl_best.pt').exists() and (out/'best.pt').exists()
    ck=torch.load(out/'last_training.pt',weights_only=False)
    assert set(('model','optimizer','scheduler','epoch')).issubset(ck)
print('TRANSFER_SMOKE_OK',flush=True)
