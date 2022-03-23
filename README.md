# Molecular Dynamics Trajectories analysis

The script will compute the Root Mean Square Deviation and the contacts plots for a molecular dynamics simulation trajectory.

## Conda environment

A [conda](https://docs.conda.io/projects/conda/en/latest/index.html) YAML environment file is provided: `references/trajectories_env.yml`. The file contains all the dependencies to run the script.
The conda environment is generated using the command:
```shell script
# create the environment
conda env create -f references/trajectories_env.yml

# activate the environment
conda activate trajectories
```

If the plots outputs are in another format than `HTML`, the chromium chromediver packages should be installed:
```shell script
sudo apt install chromium-chromedriver
```