#!/bin/bash
work_path=$(dirname $0)

file=$1
g=0
python3 docker/run_docker.py \
  --gpu_devices $g \
  --fasta_paths=$file \
  --max_template_date=2020-05-14 \
  --preset=full_dbs \

