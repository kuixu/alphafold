#!/bin/bash

# conda pkgs
conda create -n af2 python=3.8 -y
conda activate af2

conda install -y -c nvidia cudnn==8.0.4
conda install -y -c bioconda hmmer hhsuite==3.3.0 kalign2

conda install -y -c conda-forge \
      openmm=7.5.1 \
      pdbfixer \
      pip

# python pkgs
pip3 install --upgrade pip \
    && pip3 install -r ./requirements.txt \
    && pip3 install --upgrade "jax[cuda111]" -f \
    https://storage.googleapis.com/jax-releases/jax_releases.html \
    && pip3 install jaxlib==0.1.70+cuda111 -f \
    https://storage.googleapis.com/jax-releases/jax_releases.html

# work_path=/path/to/alphafold-code
work_path="$PWD"
# update openmm 
a="$(which python)"
cd "$(dirname "$(dirname "$a")")/lib/python3.8/site-packages"
patch -p0 < "$work_path/docker/openmm.patch"
