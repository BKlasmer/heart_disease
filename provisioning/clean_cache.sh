#!/bin/bash

BUILD_DIRECTORY_PATH="./build"
DIST_DIRECTORY_PATH="./dist"

# Clean python cache files in all the subdirs from the root
find .. -type d -name __pycache__ -exec rm -r {} \+ 2>&1 > /dev/null

# Clean pytest cache files in all the subdirs from the root
find .. -type d -name .pytest_cache -exec rm -r {} \+ 2>&1 > /dev/null

# Clean build/dist directories
rm -rf ${BUILD_DIRECTORY_PATH}/* ${DIST_DIRECTORY_PATH}/* ../*/*.egg-info ../*.egg-info
