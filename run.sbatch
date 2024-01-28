#!/bin/bash
#
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=weight_norm
#SBATCH --partition=dali
#SBATCH --output=weight_norm.o%j
##SBATCH --error=weight_norm.e%j
##SBATCH --mail-user=lucmon@illinois.edu
##SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:A100:1
#
# End of embedded SBATCH options
#
bash dependency.bash

. /projects/dali/spack/share/spack/setup-env.sh
spack load py-numpy/awzb36e
spack load py-torchvision/excxi5e
spack load py-scikit-learn/kivcq7b
spack load py-scipy/zdnuobl
spack load py-matplotlib/cn7no7k

python main.py
