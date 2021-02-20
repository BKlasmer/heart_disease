#!/bin/bash

VENV_CREATION_PATH_DEFAULT="./venv"

# Check if the virtual enviroment was created
if [ ! -d "${VENV_CREATION_PATH_DEFAULT}/bin" ]; then
  # Create and provisition virtual enviroment
  ./provisioning/provision_venv.sh "${VENV_CREATION_PATH_DEFAULT}"
fi

# Run tests (pass input parameters to this script to pytest too)
${VENV_CREATION_PATH_DEFAULT}/bin/pytest -v -s ./tests/ $1
