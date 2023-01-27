# Molecular Dynamics Trajectory contacts analysis

From a molecular dynamics trajectory file, the script performs a trajectory analysis to search contacts. The script 
looks for the hydrogen bonds between the atoms of two different residues. 

An hydrogen bond is defined as A-HD, where A is the acceptor heavy atom, H is the hydrogen and D is the donor heavy 
atom. An hydrogen bond is formed when A to D distance < distance cutoff and A-H-D angle > angle cutoff.
A contact is valid if the number of frames (defined by the user with --frames or on the whole data) where a contact 
is produced between 2 atoms is greater or equal to the proportion threshold of contacts.

The hydrogen bonds are represented as 2 CSV files:
   - the contacts by frame (compressed file).
   - the contacts median distance by residue.

## Conda environment

A [conda](https://docs.conda.io/projects/conda/en/latest/index.html) YAML environment file is provided: `conda_env/trajectories_env.yml`. The file contains all the dependencies to run the script.
The conda environment is generated using the command:
```shell script
# create the environment
conda env create -f conda_env/trajectories_env.yml

# activate the environment
conda activate traj
```

## Usage

The script can be tested with the test data provided in the `data` directory, which contains a trajectory file 
`traj_test.nc` with 2000 frames and the topology associated file `traj_test.parm`. The commands are:

```shell script
conda activate traj

./trajectories_contacts.py --frames 500-2000 --proportion-contacts 50.0 \
--distance-contacts 3.0 --angle-cutoff 135  --out results/traj_test --md-time "2 ns"\
--sample "trajectory test" --topology data/traj_test.parm  data/traj_test.nc

conda deactivate
```

The optional parameter used are:
- `--frames 500-2000`: selection of the frames 500 to 2000.
- `--proportion-contacts 50.0`: a contact is validated only if it is at least present in 50% of the frames 500 to 2000.
- `--distance-contacts 3.0`: maximal distance in Angstroms between 2 atoms of different residues.
- `--angle-cutoff 135`: the minimal angle in a contact between a donor/hydrogen/acceptor.
- `--md-time`: the molecular dynamics simulation duration.
- `--sample`: the sample name.

## Outputs

The script outputs are:
- a CSV file of the contacts by residue.
- a YAML file of the parameters used for this analysis. This file will be used for the script that creates the plots.
- a compressed CSV file of the contacts by frame.