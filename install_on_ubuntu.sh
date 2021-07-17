#!/bin/bash
work_path=$(dirname $0)

sudo apt-get install -y \
      build-essential \
      cmake \
      git \
      hmmer \
      kalign \
      tzdata \
      wget \

git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git 
mkdir -p ./hh-suite/build 
cd ./hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j 4 && sudo make install \
    && sudo ln -s /opt/hhsuite/bin/* /usr/bin \
cd $work_path   

conda install -y -c conda-forge \
      openmm=7.5.1 \
      pdbfixer \
      pip

wget -q -P ./alphafold/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

pip3 install --upgrade pip \
    && pip3 install -r ./requirements.txt \
    && pip3 install --upgrade "jax[cuda111]" -f \
      https://storage.googleapis.com/jax-releases/jax_releases.html

a=$(which python)
cd $(dirname $(dirname $a))/lib/python3.8/site-packages
# cd /opt/conda/lib/python3.8/site-packages
patch -p0 < $work_path/docker/openmm.patch

