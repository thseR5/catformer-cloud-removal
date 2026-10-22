#!/bin/bash
#SBATCH -J cr_eda -p rtx -A courses -q course -x cn02 -c 8 --mem=32G -t 02:00:00 -o /scratch/tadi.reddy.ug24/cloudrm/logs/eda_%j.out
cd /scratch/tadi.reddy.ug24/cloudrm
/opt/ohpc/pub/apps/miniforge3/bin/python3.12 eda.py --root data --out eda --max_locs 300
