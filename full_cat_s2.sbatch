#!/bin/bash
#SBATCH -p rtx -A courses -q course -x cn02 -J cr_cat --gres=gpu:1 -c 16 --mem=160G -t 02:00:00 -o /scratch/tadi.reddy.ug24/cloudrm/logs/train_%j.out
cd /scratch/tadi.reddy.ug24/cloudrm
/opt/ohpc/pub/apps/miniforge3/bin/python3.12 train.py --root data/asiaWest --model catformer --out runs/full_catformer_v2_s2 --epochs 40 --stage2_epochs 20 --bs 8 --seed 2
