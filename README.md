# Molecular Dynamics Trajectory contacts analysis

From molecular dynamics trajectory files (*.nc), the script performs a trajectory analysis to search contacts. It 
looks for the hydrogen bonds between the atoms of two different residues. 

An hydrogen bond is defined as A-HD, where A is the acceptor heavy atom, H is the hydrogen and D is the donor heavy 
atom. An hydrogen bond is formed when A to D distance < distance cutoff and A-H-D angle > angle cutoff.
A contact is valid if the number of frames (defined by the user with --frames or on the whole data) where a contact 
is produced between two atoms is greater or equal to the proportion threshold of contacts.

The hydrogen bonds are represented as two CSV files:
   - the median of the atoms contacts distances between two residues on the selected frames.
   - the contacts median distance of the atoms contacts between two different residues, if more than a couple of atoms 
are in contact between the two residues, the closest one will be selected.

## Conda environment

A [conda](https://docs.conda.io/projects/conda/en/latest/index.html) YAML environment file is provided: 
`conda_env/contacts_env.yml`. The file contains all the dependencies to run the script.

The conda environment is generated using the command:
```shell script
# create the environment
conda env create -f conda_env/contacts_env.yml

# activate the environment
conda activate contacts
```

## Usage

The script can be tested with the test data provided in the `tests/test_files` directory, which contains the trajectory 
files `test_data_20-frames.nc` and `test_data_20-frames_2.nc` with 20 frames and the topology associated file \
`test_data.parm`. The commands are:

```shell script
conda activate contacts

# run with 4 processors in parallel
mpirun -np 4 python ./trajectories_contacts.py --sample "test_data_20-frames" --frames test_data_20-frames.nc:5-20 \
--proportion-contacts 50.0 --distance-contacts 3.0 --angle-cutoff 135 --nanoseconds 1 \
--out results/traj_test  --topology tests/test_files/test_data.parm \
tests/test_files/test_data_20-frames.nc 

conda deactivate
```

The optional parameters used are:
- `--frames test_data_20-frames.nc:5-20`: selection of the frames 5 to 20 from the file `test_data_20-frames.nc`.
- `--proportion-contacts 50.0`: a contact is validated only if it is at least present in 50% of the frames 500 to 2000.
- `--distance-contacts 3.0`: maximal distance in Angstroms between 2 atoms of different residues.
- `--angle-cutoff 135`: the minimal angle in a contact between a donor/hydrogen/acceptor.
- `--nanoseconds`: the molecular dynamics simulation duration.

If the analysis resumes a previous analysis, use the `--resume` parameter with the path of the YAML file of the 
previous analysis as argument:

```shell script
# run with 4 processors in parallel
mpirun -np 4 python ./trajectories_contacts.py --sample "test_data_20-frames" --resume tests/expected/analysis_resumed.yaml \
--frames test_data_20-frames.nc:5-20 --proportion-contacts 50.0 --distance-contacts 3.0 \
--angle-cutoff 135 --nanoseconds 1 --out results/traj_test --topology tests/test_files/test_data.parm \
tests/test_files/test_data_20-frames_2.nc
```

## Outputs

The script outputs are:
- a YAML file of the parameters used for this analysis and the hydrogen bonds found. This file will be used for the 
script that creates the plots and can also be used to resume the analysis with new trajectory files to continue the 
molecular dynamic analysis. 
- a CSV file of the contacts with the selected frames median's distances.