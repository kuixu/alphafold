#!/bin/bash
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
file=$1
part=gpu
n=1
n1=1
name=$(basename $work_path)
srun --mpi=pmi2 --gres=gpu:${n1} \
  -p $part -n${n} \
  --ntasks-per-node=${n1} \
  -J $name -K \
  python run_alphafold.py \
  --fasta_paths=data/$file\
  |tee -a data/out/${file}.log


