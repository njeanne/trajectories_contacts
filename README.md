# Molecular Dynamics Trajectories analysis

The script will compute the Root Mean Square Deviation and the contacts plots for a molecular dynamics simulation trajectory.

## Conda environment

A [conda](https://docs.conda.io/projects/conda/en/latest/index.html) YAML environment file is provided: `conda_env/trajectories_env.yml`. The file contains all the dependencies to run the script.
The conda environment is generated using the command:
```shell script
# create the environment
conda env create -f conda_env/trajectories_env.yml

# activate the environment
conda activate trajectories
```