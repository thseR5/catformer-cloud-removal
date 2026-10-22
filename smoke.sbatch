#!/bin/bash
#SBATCH -p rtx -A courses -q course -x cn02 --gres=gpu:1 -c 16 --mem=64G -J cr_smoke -t 01:00:00 -o /scratch/tadi.reddy.ug24/cloudrm/logs/smoke_%j.out
cd /scratch/tadi.reddy.ug24/cloudrm
/opt/ohpc/pub/apps/miniforge3/bin/python3.12 train.py --root data --out runs/smoke --epochs 1 --stage2_epochs 1 --max_samples 24 --bs 4
