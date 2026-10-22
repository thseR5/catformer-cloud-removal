#!/bin/bash
#SBATCH -p rtx -A courses -q course -x cn02 --gres=gpu:1 -c 16 --mem=64G -J cr_v2 -t 12:00:00 -o /scratch/tadi.reddy.ug24/cloudrm/logs/train_%j.out
cd /scratch/tadi.reddy.ug24/cloudrm
/opt/ohpc/pub/apps/miniforge3/bin/python3.12 train.py --root data --out runs/catformer_v2 --epochs 80 --stage2_epochs 40 --bs 8
