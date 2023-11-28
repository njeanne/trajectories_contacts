#!/bin/bash

#SBATCH -J SERIAL_traj
#SBATCH --mem=96000
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-core=1
#SBATCH --time=14:00:00
#SBATCH --mail-user=jeanne.n@chu-toulouse.fr
#SBATCH --mail-type=ALL
#SBATCH --output=SERIAL_traj.%j.out

module purge
module load conda/4.9.2-silent
ulimit -s 10240

# Analysis variables
SAMPLE=HEPAC-6_RNF19A_ORF1
NB=0
TYPE=insertions
ANALYSIS=MD-1M
NANOSECONDS=2
FRAMES=$SAMPLE"_"$NB"_MD-10M.nc:10000-20000"
TMPDIR=/tmpdir/jeanne

conda activate traj

/users/p1237/jeanne/dev/trajectories_contacts/trajectories_contacts.py \
--log-level DEBUG --sample "$SAMPLE" --proportion-contacts 50.0 --distance-contacts 3.0 --angle-cutoff 135 \
--nanoseconds $NANOSECONDS --out results/serial/traj_SERIAL \
--topology "data/HEPAC-6_RNF19A_ORF1_0.parm" \
"data/HEPAC-6_RNF19A_ORF1_2000-frames.nc"

conda deactivate
