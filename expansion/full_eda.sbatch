#!/bin/bash
#SBATCH -J cr_full_eda
#SBATCH -p rtx
#SBATCH -A courses
#SBATCH -q course
#SBATCH -x cn02
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /scratch/tadi.reddy.ug24/cloudrm/logs/full_eda_%j.out
set -euo pipefail
cd /scratch/tadi.reddy.ug24/cloudrm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/opt/ohpc/pub/apps/miniforge3/bin/python3.12 expansion/full_eda.py --root data --out eda/full --workers 16
