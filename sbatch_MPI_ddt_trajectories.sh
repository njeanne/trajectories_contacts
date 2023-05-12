#!/bin/bash

#SBATCH -J MPI_traj_ddt
#SBATCH --mem=16000
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-core=1
#SBATCH --time=00:05:00
#SBATCH --mail-user=jeanne.n@chu-toulouse.fr
#SBATCH --mail-type=ALL
#SBATCH --output=MPI_traj_ddt.%j.out

echo "MPI DDT trajectories"
module purge
module load intel/18.2 intelmpi/18.2 conda/4.9.2-silent
export ALLINEA_NO_TIMEOUT
export ALLINEA_MPI_INIT=main
export ALLINEA_HOLD_MPI_INIT=1
module load arm/22.1.1
ulimit -s 10240

# variable needed to state that each mpi process use only one thread
export OMP_NUM_THREADS=1

# Analysis variables
SAMPLE=HEPAC-6_RNF19A_ORF1
NANOSECONDS=2
TMPDIR=/tmpdir/jeanne

conda activate traj

ddt --connect python3 %allinea_python_debug% /users/p1237/jeanne/dev/trajectories_contacts/trajectories_contacts_mpi.py \
--log-level DEBUG --sample "$SAMPLE" --proportion-contacts 50.0 --distance-contacts 3.0 --angle-cutoff 135 \
--nanoseconds $NANOSECONDS --out results/parallel/traj_map \
--topology "data/HEPAC-6_RNF19A_ORF1_0.parm" \
"data/HEPAC-6_RNF19A_ORF1_2000-frames.nc"

conda deactivate
