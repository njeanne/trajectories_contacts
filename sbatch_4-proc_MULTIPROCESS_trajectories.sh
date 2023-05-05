#!/bin/bash

#SBATCH -J MULTIPROC_4-proc_traj
#SBATCH --mem=16000
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-core=1
#SBATCH --time=00:05:00
#SBATCH --mail-user=jeanne.n@chu-toulouse.fr
#SBATCH --mail-type=ALL
#SBATCH --output=MULTIPROC_4-proc_traj.%j.out

module purge
module load intel/18.2 openmpi/icc/2.0.2.10 conda/4.9.2
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

srun python /users/p1237/jeanne/dev/trajectories_contacts/trajectories_contacts.py \
--log-level DEBUG --sample "$SAMPLE" --proportion-contacts 50.0 --distance-contacts 3.0 --angle-cutoff 135 \
--nanoseconds $NANOSECONDS --out results/parallel/traj \
--topology "data/HEPAC-6_RNF19A_ORF1_0.parm" \
"data/HEPAC-6_RNF19A_ORF1_2000-frames.nc"

conda deactivate
