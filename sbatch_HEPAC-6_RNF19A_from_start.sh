#!/bin/bash

#SBATCH -J HEPAC-6_RNF19A_ORF1_0_trajectories_contacts
#SBATCH --mem=96000
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --mail-user=jeanne.n@chu-toulouse.fr
#SBATCH --mail-type=ALL
#SBATCH --output=trajectories_contacts_HEPAC-6_RNF19A_0.%j.out

module purge
module load conda/4.9.2
ulimit -s 10240

# Analysis variables
SAMPLE=HEPAC-6_RNF19A_ORF1
NB=0
TYPE=insertions
ANALYSIS=MD-120M
NANOSECONDS=240
FRAMES=$SAMPLE"_"$NB"_MD-10M.nc:10000-20000"
TMPDIR=/tmpdir/jeanne

conda activate traj

/users/p1237/jeanne/dev/trajectories_contacts/trajectories_contacts.py \
--log-level DEBUG --sample "$SAMPLE" --proportion-contacts 50.0 --distance-contacts 3.0 --angle-cutoff 135 \
--nanoseconds $NANOSECONDS --frames $FRAMES --out $TMPDIR/MD/analysis/contacts/$TYPE/$ANALYSIS \
--topology $TMPDIR/MD/results/ORF1/$SAMPLE"_"$NB/$SAMPLE"_"$NB.parm \
$TMPDIR/MD/results/ORF1/$SAMPLE"_"$NB/$SAMPLE"_"$NB"_MD-10M.nc" \
$TMPDIR/MD/results/ORF1/$SAMPLE"_"$NB/$SAMPLE"_"$NB"_MD-20M.nc"

conda deactivate
