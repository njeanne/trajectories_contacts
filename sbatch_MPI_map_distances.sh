#!/bin/bash

#SBATCH -J MPI_MAP_distances
#SBATCH --mem=8000
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-core=1
#SBATCH --time=00:05:00
#SBATCH --mail-user=jeanne.n@chu-toulouse.fr
#SBATCH --mail-type=ALL
#SBATCH --output=MPI_MAP_distances.%j.out

module purge
module load intel/18.2 intelmpi/18.2 conda/4.9.2
module load arm/22.1.1
ulimit -s 10240

# variable needed to state that each mpi process use only one thread
export OMP_NUM_THREADS=1

conda activate traj

map --profile srun /users/p1237/jeanne/dev/trajectories_contacts/distances_mpi.py

conda deactivate
