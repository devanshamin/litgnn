#!/bin/bash

# Setting the script to exit as soon as any command returns a
# non-zero exit status (error occurs)
set -e

ENV_NAME="litgnn"
PYTORCH_VERSION="2.2.0" # Similar to `pyproject.toml`
# Available versions for PyTorch 2.2.0 are 'cpu', 'cu118', 'cu121'
# https://pytorch.org/get-started/previous-versions/
HARDWARE=${1:-cpu} # Default set to 'cpu'

# Check if hardware is valid
if [[ $HARDWARE != "cpu" && $HARDWARE != "cu118" && $HARDWARE != "cu121" ]];
then
    echo "Invalid hardware option! Please use one of the following: cpu, cu118, cu121"
    exit 1
fi

# Check if Operating System (OS) is valid
case $OSTYPE in
  solaris*) OSTYPE="Solaris" ;;
  darwin*)  OSTYPE="OS X" ;;
  linux*)   OSTYPE="Linux" ;;
  bsd*)     OSTYPE="BSD" ;;
  msys*)    OSTYPE="Windows" ;;
  *)        OSTYPE="Unknown: $OSTYPE" ;;
esac
if [[ $OSTYPE != "Windows" && $OSTYPE != "Linux" ]];
then 
    echo "$OSTYPE is not supported yet! Please use one the following: Windows, Linux"
    exit 1
fi

# Setup conda environment
conda create -n $ENV_NAME python=3.9 -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install requirements from `pyproject.toml` using `poetry`
poetry install
poetry run python -m pip install --upgrade pip
poetry run python -m ipykernel install --user --name=$ENV_NAME

# Install PyTorch if hardware is set to a CUDA version
# Poetry will by default install PyTorch CPU version
if [[ $HARDWARE != "cpu" ]];
then
    # Remove CPU version
    poetry run pip uninstall torch -y
    poetry run pip uninstall torch_geometric -y

    # Install GPU version
    poetry run pip install torch==$PYTORCH_VERSION --index-url https://download.pytorch.org/whl/$HARDWARE
    poetry run pip install git+https://github.com/devanshamin/pytorch_geometric.git@models/cmpnn
fi

# Install PyG additional libraries
# https://github.com/pyg-team/pytorch_geometric?tab=readme-ov-file#additional-libraries
poetry run pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-$PYTORCH_VERSION+$HARDWARE.html

echo "Installation has been successfully completed!" >&1 | tee output.log 2>&1
