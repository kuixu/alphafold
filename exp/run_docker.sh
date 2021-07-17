#!/bin/bash
work_path=$(dirname $0)
PATH="/home/jsptgpu/anaconda3/envs/af2/bin":$PATH

conda activate af2
cd /home/jsptgpu/xukui/exper/alphafold
# newgrp docker 
part=gpu
n=1
n1=1
j=$1
file=jobs/$j/$j.fasta
g=3
g=$(nvgpu available -l 1)
name=$(basename $work_path)
# srun --mpi=pmi2 --gres=gpu:${n1} \
#   -p $part -n${n} \
#   --ntasks-per-node=${n1} \
#   -J $name -K \
python3 docker/run_docker.py \
  --gpu_devices $g \
  --fasta_paths=$file \
  --max_template_date=2020-05-14 \
  --preset=full_dbs \
  # | tee -a jobs/$j/log

