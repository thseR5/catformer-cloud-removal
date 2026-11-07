#!/bin/bash
#SBATCH -J east_zero_shot
#SBATCH -p rtx
#SBATCH -A courses
#SBATCH -q course
#SBATCH -x cn02
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/tadi.reddy.ug24/cloudrm/logs/east_zero_shot_%j.out
set -euo pipefail
cd /scratch/tadi.reddy.ug24/cloudrm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/opt/ohpc/pub/apps/miniforge3/bin/python3.12 expansion/cross_region.py zero_shot --out runs/transfer_asiaeast_s0/zero_shot --workers 4 --bs 4
