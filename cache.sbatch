#!/bin/bash
#SBATCH -p rtx -A courses -q course -x cn02 -J cr_cache -c 32 --mem=64G -t 04:00:00 -o /scratch/tadi.reddy.ug24/cloudrm/logs/cache_%j.out
cd /scratch/tadi.reddy.ug24/cloudrm
/opt/ohpc/pub/apps/miniforge3/bin/python3.12 cache_clouds.py data/asiaWest 32
