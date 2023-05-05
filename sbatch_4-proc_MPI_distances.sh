#!/bin/bash

#SBATCH -J MPI_4-proc_distances
#SBATCH --mem=16000
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-core=1
#SBATCH --time=00:05:00
#SBATCH --mail-user=jeanne.n@chu-toulouse.fr
#SBATCH --mail-type=ALL
#SBATCH --output=MPI_4-proc_distances.%j.out

module purge
module load intel/18.2 openmpi/icc/2.0.2.10 conda/4.9.2
ulimit -s 10240

# variable needed to state that each mpi process use only one thread
export OMP_NUM_THREADS=1

conda activate traj

rm "pytraj_mpi.log"
srun python /users/p1237/jeanne/dev/trajectories_contacts/distances_mpi.py

conda deactivate
