#!/bin/bash
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
file=$1
# CUDA_VISIBLE_DEVICES=0
python run_alphafold.py \
  --fasta_paths=$file\
  --max_template_date=2020-05-14 \
  --preset=full_dbs \


