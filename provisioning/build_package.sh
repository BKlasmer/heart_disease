#!/bin/bash

VENV_CREATION_PATH_DEFAULT="./venv"
BUILD_DIRECTORY_PATH="./build"
DIST_DIRECTORY_PATH="./dist"

# Check if the virtual enviroment was created
if [ ! -d "${VENV_CREATION_PATH_DEFAULT}/bin" ]; then
  # Create and provisition virtual enviroment
  ./provisioning/provision_venv.sh
fi

# Following : https://bit.ly/2WyXwyM
# Install requiered packages in virtual enviroment
${VENV_CREATION_PATH_DEFAULT}/bin/pip3 install setuptools wheel

# Clean build/dist directories
rm -rf ${BUILD_DIRECTORY_PATH}/* ${DIST_DIRECTORY_PATH}/* ../*/*.egg-info ../*.egg-info

# Generate the wheel using the virtual enviroment
${VENV_CREATION_PATH_DEFAULT}/bin/python3 ./setup.py bdist_wheel
