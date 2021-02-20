#!/bin/bash

VENV_CREATION_PATH_DEFAULT="./venv"
VENV_CREATION_PATH="${VENV_CREATION_PATH_DEFAULT}"

# If no argument provided (installation path), use the default
if [ -z "$1" ]; then
  echo "No path provided for Virtual Enviroment, using default : " ${VENV_CREATION_PATH_DEFAULT}
else
  VENV_CREATION_PATH="$1"
  echo "Cleaning Virtual Enviroment in : " ${VENV_CREATION_PATH}
fi

# Clean virtual enviroment files excluding the pip config and .keep
find ${VENV_CREATION_PATH}/* -type f -not -name 'pip.conf' -not -name '.keep' -exec rm -rf {} + 2>&1 > /dev/null
find ${VENV_CREATION_PATH}/* -type d -not -name 'pip.conf' -not -name '.keep' -exec rm -rf {} + 2>&1 > /dev/null
