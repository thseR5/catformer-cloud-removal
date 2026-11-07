"""Full EDA over every indexed location in the downloaded West and East regions."""
import argparse, csv, json, os, re, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import index_files

def summarize_region(region, root, workers, outdir):
    root = Path(root) / region
    idx = index_files(str(root))
    keys = sorted(idx)
    print(f"{region}: indexed {len(keys)} locations", flush=True)
    tasks=[]
    for key in keys:
        v=idx[key]
        paired=sorted(set(v['s1']) & set(v['s2']))
        tasks.append((key,v,paired,root,region))

    def one(task):
        key,v,paired,root,region=task
        s2_cov=[]; s2_cov50=[]; nodata=[]; shape_counts=Counter(); s1_nan=0
        s2_means=[]; s1_means=[]; errors=[]
        for t in sorted(v['s2']):
            p,date=v['s2'][t]
            try:
                cp=np.load(p[:-4]+'_cld.npy', mmap_mode='r')
                if cp.ndim==3: cp=cp.squeeze()
                c=np.asarray(cp,dtype=np.float32)/255.0
                s2_cov.append(float((c>.4).mean())); s2_cov50.append(float((c>.5).mean()))
                with rasterio.open(p) as ds:
                    a=ds.read(out_dtype='float32')
                shape_counts['x'.join(map(str,a.shape))]+=1
                nodata.append(float((a==0).all(axis=0).mean()))
                s2_means.append(np.nanmean(a[:,::32,::32],axis=(1,2)).astype(float).tolist())
            except Exception as e: errors.append('s2:'+type(e).__name__)
        for t in sorted(v['s1']):
            p,date=v['s1'][t]
            try:
                with rasterio.open(p) as ds: a=ds.read(out_dtype='float32')
                shape_counts['x'.join(map(str,a.shape))]+=1
                s1_nan+=int((~np.isfinite(a)).sum())
                s1_means.append(np.nanmean(a[:,::32,::32],axis=(1,2)).astype(float).tolist())
            except Exception as e: errors.append('s1:'+type(e).__name__)
        def avg(x): return float(np.mean(x)) if x else None
        return dict(region=region,roi=key[0],tile=key[1],patch=key[2],
          s2_dates=len(v['s2']),s1_dates=len(v['s1']),paired_dates=len(paired),
          cloud_fraction_gt04_mean=avg(s2_cov),cloud_fraction_gt05_mean=avg(s2_cov50),
          s2_zero_pixel_fraction_mean=avg(nodata),s1_nonfinite_values=s1_nan,
          s2_band_means_dn=np.nanmean(np.asarray(s2_means),axis=0).tolist() if s2_means else [],
          s1_band_means_db=np.nanmean(np.asarray(s1_means),axis=0).tolist() if s1_means else [],
          shapes=dict(shape_counts),errors=dict(Counter(errors)))

    rows=[]; start=time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,t) for t in tasks]
        for n,f in enumerate(as_completed(futures),1):
            rows.append(f.result())
            if n%100==0 or n==len(tasks):
                print(f"{region}: completed {n}/{len(tasks)} in {(time.time()-start)/60:.1f} min",flush=True)
    rows.sort(key=lambda r:(r['roi'],r['tile'],r['patch']))
    # Keep frame-level cloud results separately. Maps are cached, so a second streaming pass
    # avoids retaining image tensors and permits exact date-level cloud summaries.
    cloud_rows=[]; cloud_by_location=defaultdict(list)
    for key in keys:
        for t,(p,date) in sorted(idx[key]['s2'].items()):
            try:
                c=np.load(p[:-4]+'_cld.npy',mmap_mode='r').astype(np.float32)/255.0
                cov04=float((c>.4).mean()); cov05=float((c>.5).mean())
                cloud_rows.append((region,key[0],key[1],key[2],t,date,cov04,cov05))
                cloud_by_location[key].append(cov04)
            except Exception: pass
    summary={"region":region,"locations_indexed":len(keys),
       "locations_complete_30":sum(r['s1_dates']==30 and r['s2_dates']==30 for r in rows),
       "paired_date_count_distribution":dict(Counter(str(r['paired_dates']) for r in rows)),
       "s2_frames":sum(r['s2_dates'] for r in rows),"s1_frames":sum(r['s1_dates'] for r in rows),
       "mean_cloud_fraction_gt04":float(np.mean([x[6] for x in cloud_rows])),
       "median_cloud_fraction_gt04":float(np.median([x[6] for x in cloud_rows])),
       "fraction_frames_clear_lt5pct_gt04":float(np.mean([x[6]<.05 for x in cloud_rows])),
       "fraction_frames_overcast_gt90pct_gt04":float(np.mean([x[6]>.90 for x in cloud_rows])),
       "mean_cloud_fraction_gt05":float(np.mean([x[7] for x in cloud_rows])),
       "fraction_frames_clear_lt5pct_gt05":float(np.mean([x[7]<.05 for x in cloud_rows])),
       "fraction_frames_overcast_gt90pct_gt05":float(np.mean([x[7]>.90 for x in cloud_rows])),
       "locations_without_any_frame_below5pct_cloud":sum(not any(c<.05 for c in cloud_by_location[(r['roi'],r['tile'],r['patch'])]) for r in rows),
       "mean_s2_zero_pixel_fraction":float(np.nanmean([r['s2_zero_pixel_fraction_mean'] for r in rows])),
       "locations_with_s2_read_errors":sum(bool(r['errors']) for r in rows),
       "s1_nonfinite_values_total":sum(r['s1_nonfinite_values'] for r in rows),
       "band_names_s2":["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"],
       "mean_s2_band_dn_all_locations":np.nanmean(np.asarray([r['s2_band_means_dn'] for r in rows if r['s2_band_means_dn']]),axis=0).tolist(),
       "mean_s1_band_db_all_locations":np.nanmean(np.asarray([r['s1_band_means_db'] for r in rows if r['s1_band_means_db']]),axis=0).tolist(),
       "date_range_s2":[min(x[5] for x in cloud_rows),max(x[5] for x in cloud_rows)],
       "shapes":dict(sum((Counter(r['shapes']) for r in rows),Counter()))}
    Path(outdir).mkdir(parents=True,exist_ok=True)
    with open(Path(outdir)/f'{region}_locations.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    with open(Path(outdir)/f'{region}_cloud_frames.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['region','roi','tile','patch','t','date','cloud_fraction_gt04','cloud_fraction_gt05']);w.writerows(cloud_rows)
    json.dump(summary,open(Path(outdir)/f'{region}_summary.json','w'),indent=2)
    print(json.dumps(summary,indent=2),flush=True)
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='data');ap.add_argument('--out',default='eda/full');ap.add_argument('--workers',type=int,default=12)
    a=ap.parse_args();summaries=[]
    for region in ('asiaWest','asiaEast'): summaries.append(summarize_region(region,a.root,a.workers,a.out))
    combined={'scope':'all extracted asiaWest and asiaEast locations; s2cloudless cached probabilities; all indexed frames','regions':summaries}
    json.dump(combined,open(Path(a.out)/'full_summary.json','w'),indent=2)
    print('FULL_EDA_DONE',flush=True)
if __name__=='__main__':main()
