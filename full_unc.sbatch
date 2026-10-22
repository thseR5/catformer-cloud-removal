#!/bin/bash
#SBATCH -p rtx -A courses -q course -x cn02 -J cr_unc --gres=gpu:1 -c 16 --mem=160G -t 10:00:00 -o /scratch/tadi.reddy.ug24/cloudrm/logs/train_%j.out
cd /scratch/tadi.reddy.ug24/cloudrm
/opt/ohpc/pub/apps/miniforge3/bin/python3.12 train.py --root data/asiaWest --model uncrtaints --out runs/full_uncrtaints_s0 --epochs 40 --bs 8 --lr 1e-3 --seed 0
