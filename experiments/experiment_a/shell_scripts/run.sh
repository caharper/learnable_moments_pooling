#!/usr/bin/env zsh
#SBATCH -J aperture
#SBATCH -o aperture_%j.out
#SBATCH -c 128 --mem=256G      # requested partition
#SBATCH --nodes=1
#SBATCH -G 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00              # maximum runtime 2 days


# Preprocess the data
# Flags:
# --no-container-remap-root : Set actual user inside container (not root)
# --container-image         : Source container image
# --container-save          : Destination container image
# --no-container-entrypoint : Do not use image entrypoint
# --container-mounts        : Mount NeMo repo as workspace inside container
# bash -l -c ./reinstall.sh : Run the NeMo reinstall.sh script path already
#                             in NeMo directory due to --container-mounts

# Replace aperture-layers with your repo name

chmod +x /aperture-layers/experiments/inaturalist2018/entrypoint.sh
srun\
    --no-container-entrypoint\
    --container-image /work/users/caharper/containers/tensorflow+tensorflow+2.13.0-gpu.sqsh\
    --container-mounts="${HOME}"/"aperture-layers"/:/"aperture-layers",/"work"/users/caharper/datasets/images:/aperture-layers/data\
    --container-workdir /aperture-layers\
    bash -c /aperture-layers/experiments/inaturalist2018/entrypoint.sh


# docker pull tensorflow/tensorflow:2.13.0-gpu
# docker pull tensorflow/tensorflow:2.13.0-gpu
# enroot import docker://tensorflow/tensorflow:2.13.0-gpu
