#!/bin/bash

#SBATCH -J SERIAL_distances
#SBATCH --mem=16000
#SBATCH -N 1
#SBATCH --time=00:05:00
#SBATCH --mail-user=jeanne.n@chu-toulouse.fr
#SBATCH --mail-type=ALL
#SBATCH --output=SERIAL_distances.%j.out

module purge
module load conda/4.9.2
ulimit -s 10240

conda activate traj

python /users/p1237/jeanne/dev/trajectories_contacts/distances_serial.py

conda deactivate
