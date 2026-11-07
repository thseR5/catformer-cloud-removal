"""Fixed geographic transfer experiment. Run from the project root.

prepare -> zero-shot -> finetune / ssl_finetune. Original code and weights are
untouched. SSL reconstructs masked observed training inputs, never test targets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from data import index_files, read, norm_s1, norm_s2, cached_cloud, nodata_mask
from model import CATFormer, loss_fn

NAMES = ('MAE', 'PSNR', 'SSIM_rgb', 'SAM')


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def quality_record(item):
    """Choose exactly one valid target per location, fixed before any model runs."""
    region, key, v, root = item
    try:
        ts = sorted(set(v['s1']) & set(v['s2']))
        if len(ts) < 4:
            return None, 'fewer_than_four_paired_dates'
        cov, valid = {}, []
        for t in ts:
            x = read(v['s2'][t][0])
            if x.shape != (13, 256, 256):
                return None, 'unexpected_optical_shape'
            if not np.isfinite(x).all():
                continue
            missing = nodata_mask(x)
            c = cached_cloud(v['s2'][t][0])
            if c.shape != (256, 256) or not np.isfinite(c).all():
                return None, 'invalid_cloud_cache'
            cov[t] = float((c > .4).mean())
            if not missing.any():
                valid.append(t)
        if not valid:
            return None, 'no_complete_target'
        target = min(valid, key=lambda t: (cov[t], t))
        # Ground truth is a low-cloud reference, not guaranteed perfectly clear.
        if cov[target] > .05:
            return None, 'no_target_under_five_percent_cloud'
        d = lambda t: np.datetime64(v['s2'][t][1])
        candidates = []
        for t in ts:
            if t == target:
                continue
            x, s = read(v['s2'][t][0]), read(v['s1'][t][0])
            if s.shape != (2, 256, 256) or not np.isfinite(s).all() or not np.isfinite(x).all():
                continue
            if nodata_mask(x).mean() >= .5:
                continue
            candidates.append(t)
        if len(candidates) < 3:
            return None, 'fewer_than_three_valid_inputs'
        ins = sorted(sorted(candidates, key=lambda t: (abs(int((d(t) - d(target)).astype(int))), t))[:3])
        rel = lambda p: str(Path(p).relative_to(root))
        return dict(region=region, key=list(key), s2=[rel(v['s2'][t][0]) for t in ins],
                    s1=[rel(v['s1'][t][0]) for t in ins], target=rel(v['s2'][target][0]),
                    tpos=[float((d(t) - d(target)).astype(int) / 30) for t in ins],
                    target_cloud_fraction=cov[target]), None
    except Exception as e:
        return None, f'read_failure:{type(e).__name__}'


def prepare(args):
    from collections import Counter
    root = Path(args.root)
    original = json.loads(Path(args.source_split).read_text())
    index = {r: index_files(str(root / r)) for r in ('asiaWest', 'asiaEast')}
    if not index['asiaEast']:
        raise RuntimeError('asiaEast is empty')
    rois = sorted({k[0] for k in index['asiaEast']})
    if len(rois) < 3:
        raise RuntimeError('Need at least three target ROIs for geographic train/val/test')
    random.Random(1234).shuffle(rois)
    ntest = max(1, round(len(rois) * .2))
    nval = max(1, round(len(rois) * .15))
    groups = dict(test=rois[:ntest], val=rois[ntest:ntest+nval], train=rois[ntest+nval:])
    split = dict(asiaWest=original, asiaEast=groups)
    tasks = [(r, k, v, root) for r, ix in index.items() for k, v in sorted(ix.items())]
    records, bad = [], Counter()
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (rec, reason) in enumerate(ex.map(quality_record, tasks)):
            if rec is None:
                bad[reason] += 1
            else:
                r, key = rec['region'], rec['key']
                match = [s for s, values in split[r].items()
                         if (key[:2] in values if r == 'asiaWest' else key[0] in values)]
                if len(match) != 1:
                    raise RuntimeError(f'Unassigned/ambiguous record {key}')
                rec['split'] = match[0]
                records.append(rec)
            if i % 500 == 0:
                print('AUDIT', i, 'accepted', len(records), 'excluded', dict(bad), flush=True)
    counts = Counter((r['region'], r['split']) for r in records)
    for region in index:
        for phase in ('train', 'val', 'test'):
            if counts[(region, phase)] == 0:
                raise RuntimeError(f'Empty {region}/{phase} after quality filtering')
    # Detect exact patch overlap across geographic splits (georeferenced TIFFs).
    import rasterio
    from rasterio.warp import transform_bounds
    footprints = []
    for r in records:
        with rasterio.open(root / r['target']) as f:
            if f.crs is None:
                raise RuntimeError('Missing CRS: cannot verify geographic separation')
            bounds = transform_bounds(f.crs, 'EPSG:4326', *f.bounds)
        footprints.append((r['region'], r['split'], r['key'], bounds))
    overlaps = []
    for i, (region, phase, key, a) in enumerate(footprints):
        for r2, p2, k2, b in footprints[:i]:
            if phase == p2 and region == r2:
                continue
            # Existing West splits are historical; audit new East splits and West/East.
            if region == r2 == 'asiaWest':
                continue
            if min(a[2], b[2]) - max(a[0], b[0]) > 1e-7 and min(a[3], b[3]) - max(a[1], b[1]) > 1e-7:
                overlaps.append([region, phase, key, r2, p2, k2])
                if len(overlaps) >= 20:
                    break
        if len(overlaps) >= 20:
            break
    audit = dict(counts={f'{r}/{s}': n for (r, s), n in counts.items()},
                 exclusions=dict(bad), geographic_overlap_examples=overlaps,
                 protocol='3 valid nearest actual-date paired inputs; <=5% cloudy complete reference; ROI-disjoint East split',
                 split=split, checkpoint_sha256=digest(args.checkpoint))
    save_json(Path(args.out) / 'audit.json', audit)
    if overlaps:
        raise RuntimeError('Geographic overlap found; audit saved. Training blocked pending split correction.')
    save_json(Path(args.out) / 'manifest.json', dict(records=records, audit=audit))
    print('PREPARE_DONE', audit, flush=True)


class Samples(Dataset):
    def __init__(self, root, records):
        self.root, self.records = Path(root), records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        xs = [read(self.root / p) for p in r['s2']]
        cm = [np.maximum(cached_cloud(str(self.root / p)), nodata_mask(x)) for p, x in zip(r['s2'], xs)]
        return dict(s2=torch.from_numpy(np.stack([norm_s2(x) for x in xs])),
                    s1=torch.from_numpy(np.stack([norm_s1(read(self.root / p)) for p in r['s1']])),
                    cloud=torch.from_numpy(np.stack(cm)[:, None]),
                    tpos=torch.tensor(r['tpos'], dtype=torch.float32),
                    y=torch.from_numpy(norm_s2(read(self.root / r['target']))),
                    index=i)


def loader(root, records, args, shuffle=False):
    return DataLoader(Samples(root, records), batch_size=args.bs, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=args.workers > 0)


def to_device(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def forward(m, b):
    return m(b['s2'], b['s1'], b['cloud'], b['tpos'])


def scores(p, y):
    rows = []
    for a, b in zip(p.detach().float().cpu().numpy(), y.detach().float().cpu().numpy()):
        cos = (a*b).sum(0) / (np.linalg.norm(a, axis=0)*np.linalg.norm(b, axis=0)+1e-8)
        rows.append([float(np.abs(a-b).mean()), float(-10*np.log10(max(float(((a-b)**2).mean()), 1e-10))),
                     float(structural_similarity(a[[3,2,1]], b[[3,2,1]], channel_axis=0, data_range=1)),
                     float(np.degrees(np.arccos(cos.clip(-1, 1))).mean())])
    return rows


def summarize(rows):
    return dict(zip(NAMES, np.asarray(rows).mean(0).tolist()))


@torch.no_grad()
def evaluate(m, dl, dev, collect=False):
    m.eval()
    rows, base, per_patch = [], [], []
    for raw in dl:
        b = to_device(raw, dev)
        p = forward(m, b)[0]
        ss = scores(p, b['y'])
        rows.extend(ss)
        chosen = b['cloud'].mean((2,3,4)).argmin(1)
        base.extend(scores(b['s2'][torch.arange(len(chosen), device=dev), chosen], b['y']))
        if collect:
            for idx, sc in zip(raw['index'].tolist(), ss):
                r = dl.dataset.records[idx]
                per_patch.append(dict(region=r['region'], key=r['key'], **dict(zip(NAMES, sc))))
    return dict(model=summarize(rows), least_cloudy=summarize(base), n=len(rows), per_patch=per_patch)


def ssl_example(b):
    """Mask every optical date at shared blocks; reconstruct a chosen observed date.
    Only low-cloud, nonmissing pixels from that date contribute to loss.
    The supervised reference y is not used here.
    """
    B, T, _, H, W = b['s2'].shape
    ii = torch.arange(B, device=b['s2'].device)
    ti = (b['cloud'] < .1).float().mean((2,3,4)).argmax(1)
    target = b['s2'][ii, ti].clone()
    clear = (b['cloud'][ii, ti] < .1) & (target.sum(1, keepdim=True) > 0)
    mask = F.interpolate((torch.rand(B, 1, max(1,H//32), max(1,W//32), device=target.device) < .5).float(),
                         size=(H,W), mode='nearest')
    hidden = mask[:, None]
    altered = dict(b)
    altered['s2'] = b['s2'] * (1-hidden)
    altered['cloud'] = torch.maximum(b['cloud'], hidden)
    altered['tpos'] = b['tpos'] - b['tpos'][ii, ti, None]
    return altered, target, clear.float()*mask


def train_epochs(model, train_dl, val_dl, args, out, ssl=False):
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
    epochs = args.ssl_epochs if ssl else args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best, hist = None, []
    bestpath = out / ('ssl_best.pt' if ssl else 'best.pt')
    # Identity baseline may be best: do not force a destructive update.
    baseline = evaluate(model, val_dl, dev)['model']
    best = dict(epoch=-1, **baseline)
    torch.save(model.state_dict(), bestpath)
    def masked_val():
        model.eval(); total = denom = 0.
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev.type == 'cuda' else []):
            torch.manual_seed(447)
            for raw in val_dl:
                a, y, mask = ssl_example(to_device(raw, dev))
                p = forward(model, a)[0]
                total += float(((p-y).abs()*mask).sum())
                denom += float(mask.sum())*13
        if denom == 0:
            raise RuntimeError('No clear masked validation pixels for SSL')
        return total/denom
    if ssl:
        best = dict(epoch=-1, masked_MAE=masked_val())
    for ep in range(epochs):
        start = time.time(); model.train(); numerator = valid_steps = 0.
        for raw in train_dl:
            b = to_device(raw, dev)
            if ssl:
                b, y, mask = ssl_example(b)
                if mask.sum() < 1:
                    continue
            else:
                # Joint spatial augmentation, including optical, radar, mask and reference.
                k, flip = random.randrange(4), random.random() < .5
                for name in ('s2', 's1', 'cloud', 'y'):
                    b[name] = torch.rot90(b[name], k, (-2,-1))
                    if flip: b[name] = b[name].flip(-1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type=='cuda'):
                mean, lv, _ = forward(model, b)
            if ssl:
                loss = ((mean.float()-y).abs()*mask).sum()/(13*mask.sum())
            else:
                loss = loss_fn(mean.float(), lv.float(), b['y'], b['cloud'].min(1).values)[0]
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; stopping to preserve checkpoint')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); numerator += float(loss.detach()); valid_steps += 1
        if not valid_steps:
            raise RuntimeError('No valid optimization batches')
        sched.step()
        if ssl:
            metrics = dict(masked_MAE=masked_val())
            improved = metrics['masked_MAE'] < best['masked_MAE']
        else:
            metrics = evaluate(model, val_dl, dev)['model']
            improved = metrics['PSNR'] > best['PSNR']
        row = dict(epoch=ep, loss=numerator/valid_steps, seconds=time.time()-start, **metrics)
        hist.append(row); print('SSL' if ssl else 'FINETUNE', json.dumps(row), flush=True)
        if improved:
            best = row
            torch.save(model.state_dict(), bestpath)
        # Full optimizer state allows future exact continuation from the last epoch.
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                        epoch=ep, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                        python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
                   out / ('ssl_last_training.pt' if ssl else 'last_training.pt'))
        save_json(out / ('ssl_history.json' if ssl else 'history.json'), dict(best=best, history=hist))
    model.load_state_dict(torch.load(bestpath, map_location=dev, weights_only=True))
    return best


@torch.no_grad()
def embeddings(model, root, records, args, out):
    """Spatial and temporal mean of the final per-date encoder features."""
    selected = []
    for region in ('asiaWest','asiaEast'):
        for split in ('train','val','test'):
            rs = [r for r in records if r['region']==region and r['split']==split]
            selected.extend(sorted(rs, key=lambda r: tuple(r['key']))[:args.embedding_limit])
    features, labels, captured = [], [], []
    hook = model.enc[-1].register_forward_hook(lambda module, inputs, value: captured.append(value.detach()))
    model.eval(); dev = next(model.parameters()).device
    try:
        for raw in loader(root, selected, args):
            b = to_device(raw, dev); captured.clear(); forward(model,b)
            B,T = b['s2'].shape[:2]
            z = captured[-1].float().mean((-2,-1)).reshape(B,T,-1).mean(1).cpu().numpy()
            features.extend(z)
            for idx in raw['index'].tolist():
                r=selected[idx]
                labels.append(dict(region=r['region'], split=r['split'], key=r['key']))
    finally:
        hook.remove()
    z=np.asarray(features)
    np.savez_compressed(out/'embeddings.npz', features=z, region=np.array([r['region'] for r in labels]),
                        split=np.array([r['split'] for r in labels]))
    save_json(out/'embedding_labels.json',labels)


def run(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'Refusing to overwrite nonempty run directory {out}')
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    manifest=json.loads(Path(args.manifest).read_text())
    if digest(args.checkpoint) != manifest['audit']['checkpoint_sha256']:
        raise RuntimeError('Source checkpoint changed since preparation')
    records=manifest['records']
    get=lambda region, split: [r for r in records if r['region']==region and r['split']==split]
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type != 'cuda' and not args.allow_cpu:
        raise RuntimeError('GPU unavailable; refusing an unexpectedly slow training job')
    model=CATFormer().to(dev)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev, weights_only=True), strict=True)
    save_json(out/'config.json',dict(args=vars(args), source_sha256=digest(args.checkpoint),
                                   manifest_sha256=digest(args.manifest), protocol=manifest['audit']['protocol']))
    if args.mode != 'zero_shot':
        target=get('asiaEast','train')
        source=get('asiaWest','train')
        rng=random.Random(919)
        n=min(len(source), round(len(target)*args.replay_fraction/(1-args.replay_fraction)))
        source=rng.sample(source,n)
        mixed=target+source
        save_json(out/'training_locations.json',[dict(region=r['region'],key=r['key']) for r in mixed])
        tr=loader(args.root,mixed,args,shuffle=True)
        va=loader(args.root,get('asiaEast','val'),args)
        if args.mode == 'ssl_finetune':
            # SSL validation uses masked observed inputs in val, not supervised references.
            train_epochs(model,tr,va,args,out,ssl=True)
            # Reset supervised random sequence identically for the fine-tuning comparison.
            torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
        train_epochs(model,tr,va,args,out)
    results={}
    for region in ('asiaWest','asiaEast'):
        results[region]=evaluate(model,loader(args.root,get(region,'test'),args),dev,collect=True)
        print('TEST',region,json.dumps({k:v for k,v in results[region].items() if k!='per_patch'}),flush=True)
    save_json(out/'results.json',results)
    embeddings(model,args.root,records,args,out)
    print('EXPERIMENT_DONE',args.mode,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','zero_shot','finetune','ssl_finetune'])
    p.add_argument('--root',default='data')
    p.add_argument('--out',required=True)
    p.add_argument('--manifest',default='expansion/prepared/manifest.json')
    p.add_argument('--source-split',default='runs/full_catformer_v2_s0/split.json')
    p.add_argument('--checkpoint',default='runs/full_catformer_v2_s0/best.pt')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--bs',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--ssl-epochs',type=int,default=8)
    p.add_argument('--replay-fraction',type=float,default=.25)
    p.add_argument('--embedding-limit',type=int,default=100)
    p.add_argument('--allow-cpu',action='store_true')
    args=p.parse_args()
    if not 0 <= args.replay_fraction < 1:
        p.error('replay fraction must be in [0,1)')
    torch.set_num_threads(4)
    prepare(args) if args.mode=='prepare' else run(args)

if __name__=='__main__':
    main()
