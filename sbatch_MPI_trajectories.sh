#!/bin/bash

#SBATCH -J MPI_traj
#SBATCH --mem=16000
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-core=1
#SBATCH --time=00:05:00
#SBATCH --mail-user=jeanne.n@chu-toulouse.fr
#SBATCH --mail-type=ALL
#SBATCH --output=MPI_traj.%j.out

module purge
module load intel/18.2 intelmpi/18.2 conda/4.9.2-silent
ulimit -s 10240

# variable needed to state that each mpi process use only one thread
export OMP_NUM_THREADS=1

# Analysis variables
SAMPLE=HEPAC-6_RNF19A_ORF1
NB=0
TYPE=insertions
ANALYSIS=MD-1M
NANOSECONDS=2
FRAMES=$SAMPLE"_"$NB"_MD-10M.nc:10000-20000"
TMPDIR=/tmpdir/jeanne

conda activate traj

source /usr/local/intel/2018.2.046/compilers_and_libraries/../itac/2018.2.020/bin/itacvars.sh

srun -python /users/p1237/jeanne/dev/trajectories_contacts/trajectories_contacts_mpi.py \
--log-level DEBUG --sample "$SAMPLE" --proportion-contacts 50.0 --distance-contacts 3.0 --angle-cutoff 135 \
--nanoseconds $NANOSECONDS --out results/parallel/traj_MPI \
--topology "data/HEPAC-6_RNF19A_ORF1_0.parm" \
"data/HEPAC-6_RNF19A_ORF1_2000-frames.nc"

conda deactivate
