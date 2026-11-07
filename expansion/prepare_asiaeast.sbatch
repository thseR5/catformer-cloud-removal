#!/bin/bash
#SBATCH -J east_prepare
#SBATCH -p rtx
#SBATCH -A courses
#SBATCH -q course
#SBATCH -x cn02
#SBATCH -c 32
#SBATCH --mem=80G
#SBATCH -t 08:00:00
#SBATCH -o /scratch/tadi.reddy.ug24/cloudrm/logs/east_prepare_%j.out
set -euo pipefail
cd /scratch/tadi.reddy.ug24/cloudrm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/opt/ohpc/pub/apps/miniforge3/bin/python3.12
# A failed checksum or tar extraction prevents all dependent training jobs.
test -f expansion/asiaeast_checksums.txt
[ "$(grep -c ': OK$' expansion/asiaeast_checksums.txt)" -eq 2 ]
mkdir -p data
tar xzf raw/s1_asiaEast.tar.gz -C data
tar xzf raw/s2_asiaEast.tar.gz -C data
$P cache_clouds.py data/asiaEast 32 > logs/east_cloud_cache.log 2>&1
tail -n 1 logs/east_cloud_cache.log | grep -Eq 'CACHE_DONE files [1-9][0-9]* unreadable 0$'
$P expansion/cross_region.py prepare --out expansion/prepared --workers 16
