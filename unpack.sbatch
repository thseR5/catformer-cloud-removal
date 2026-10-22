#!/bin/bash
#SBATCH -J cr_unpack -p rtx -A courses -q course -x cn02 -c 8 --mem=16G -t 03:00:00 -o /scratch/tadi.reddy.ug24/cloudrm/logs/unpack_%j.out
cd /scratch/tadi.reddy.ug24/cloudrm/raw
(grep asiaWest checksums.sha512 | sha512sum -c - > checksum_result.txt 2>&1; echo CHECKSUM_DONE) &
mkdir -p ../data
(tar xzf s1_asiaWest.tar.gz -C ../data && echo S1_UNPACKED || echo S1_UNPACK_FAIL) &
(tar xzf s2_asiaWest.tar.gz -C ../data && echo S2_UNPACKED || echo S2_UNPACK_FAIL) &
wait
cat checksum_result.txt
cd ../data; for s in S1 S2; do echo $s $(find asiaWest -path "*/$s/*" -name "*.tif" | wc -l); done
echo ROIs $(ls asiaWest | wc -l) tiles $(ls -d asiaWest/*/* | wc -l)
echo ALL_DONE
