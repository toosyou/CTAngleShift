#!/bin/bash

# Install the dependencies by conda
conda install -y -n tomopy --channel conda-forge tomopy "libtomo=*=cuda*"
conda install -y -n tomopy --channel astra-toolbox astra-toolbox

conda activate tomopy

# Install the dependencies by pipenv
pipenv --python=$(conda run which python) --site-packages
pipenv sync
