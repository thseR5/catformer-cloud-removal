#!/bin/bash
#SBATCH -J east_summary
#SBATCH -p rtx
#SBATCH -A courses
#SBATCH -q course
#SBATCH -x cn02
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 00:20:00
#SBATCH -o /scratch/tadi.reddy.ug24/cloudrm/logs/east_summary_%j.out
set -euo pipefail
cd /scratch/tadi.reddy.ug24/cloudrm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/opt/ohpc/pub/apps/miniforge3/bin/python3.12 expansion/summarize_transfer.py
