#!/bin/bash

# work_path=/path/to/alphafold-code
work_path=$(PWD)
echo $work_path


# apt tools 
sudo apt-get install -y \
      build-essential \
      cmake \
      git \
      hmmer \
      kalign \
      tzdata \
      wget \


# conda pkgs
conda create -n af2 python=3.8 -y
conda activate af2

conda install -y -c conda-forge \
      openmm=7.5.1 \
      pdbfixer \
      pip

# python pkgs
pip3 install --upgrade pip \
    && pip3 install -r ./requirements.txt \
    && pip3 install --upgrade "jax[cuda111]" -f \
      https://storage.googleapis.com/jax-releases/jax_releases.html

# hh-suite
git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git 
mkdir -p ./hh-suite/build 
cd ./hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j 4 && sudo make install \
    && sudo ln -s /opt/hhsuite/bin/* /usr/bin \
cd $work_path   

# update openmm 
a=$(which python)
cd $(dirname $(dirname $a))/lib/python3.8/site-packages
patch -p0 < $work_path/docker/openmm.patch

