#!/bin/bash

VENV_CREATION_PATH_DEFAULT="./venv"
VENV_CREATION_PATH="${VENV_CREATION_PATH_DEFAULT}"
REQUIREMENTS_FILE_PATH="./requirements.txt"

# Delete existing cached egg and virtual environment
rm -rf *.egg-info
rc=$?
if [ $rc -gt 0 ]; then
    echo "Failed to delete the egg files!"
    exit $rc
fi

# Clean virtual environment except for .keep and pip.conf file.
./provisioning/clean_venv.sh
rc=$?
if [ $rc -gt 0 ]; then
    echo "Clean virtual environment script failed."
    exit $rc
fi


# If no argument provided (installation output), use the default
if [ -z "$1" ]; then
  echo "No path provided for Virtual Environment, using default : " ${VENV_CREATION_PATH_DEFAULT}
else
  VENV_CREATION_PATH="$1"
  echo "Creating Virtual Environment in : " ${VENV_CREATION_PATH}
fi

# Create Virtual enviroment with Python3
python3 -m virtualenv -p python3 "${VENV_CREATION_PATH}"
rc=$?
if [ $rc -gt 0 ]; then
    echo "Creating the virtual environment failed."
    exit $rc
fi

# upgrade pip in the venv
${VENV_CREATION_PATH}/bin/python3 -m pip install --upgrade pip

# Install requirements
${VENV_CREATION_PATH}/bin/pip3 install -r ${REQUIREMENTS_FILE_PATH}
rc=$?
if [ $rc -gt 0 ]; then
    echo "Installing the requirements failed."
    exit $rc
fi
