#!/bin/bash

echo $(ls)
cd /aperture-layers

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py --user


python3 -m pip install --upgrade pip
sudo apt-get install python3-setuptools
apt-get update && apt-get install
pip3 install bocas pandas tabulate termcolor keras_cv tensorflow_datasets keras-core
pip3 install -U setuptools
pip install --upgrade setuptools
echo $(ls)
python3 -U setup.py develop

cd experiments/experiment_a
# python3 ../../setup.py develop
# echo $(ls)
# export PYTHONPATH=$PYTHONPATH:/PUF-Modelling-and-ML

# NCCL_VERSION=$(find /usr -name "libnccl.so*" -exec sh -c 'echo {} | sed -r "s/^.*\.so\.//" ' \; | head -n1)
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl-$NCCL_VERSION/lib

python3 -m bocas.launch run.py --task run.py --config configs/v0.1/sweep-sizes.py
python ./scripts/aggregate_results.py
# python ./scripts/plot_results.py