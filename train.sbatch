#!/bin/bash
#SBATCH -J cr_train -p rtx -A courses -q course -x cn02 --gres=gpu:1 -c 16 --mem=64G -t 10:00:00 -o /scratch/tadi.reddy.ug24/cloudrm/logs/train_%j.out
cd /scratch/tadi.reddy.ug24/cloudrm
nvidia-smi --query-gpu=name,memory.total --format=csv
/opt/ohpc/pub/apps/miniforge3/bin/python3.12 train.py --root data --out runs/catformer_v1_s2cl --epochs 80 --bs 8
