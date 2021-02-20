#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import setuptools
from os import path

thisScriptLocation = path.abspath(path.dirname(__file__))
with open(path.join(thisScriptLocation, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

# Based on https://bit.ly/2WtuQlU

setuptools.setup(
    name="heart_disease",
    description="Heart Disease Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BKlasmer/heart_disease",
    # Exclude folder, subfolders and files
    packages=setuptools.find_packages(
        exclude=[
            "*.test",
            "*.test.*",
            "test.*",
            "test",
            ".github",
            "provisioning",
            "venv",
            "notebooks",
            "*.ipynb",
        ]
    ),
    # Force Python 3.6+ and 3.8- (some packages are not compatible)
    python_requires=">=3.6, <3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS",
    ],
)
