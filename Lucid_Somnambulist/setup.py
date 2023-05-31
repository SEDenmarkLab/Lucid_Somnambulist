"""
    This module enables `pip install ./`
    Do not run this script standalone.
    Refer to package readme for further details.
"""

from setuptools import setup, find_packages
from glob import glob
import os

### Add dependencies later; this is not done as of 9-9-2022

### Need to make a conda install so that crest/xtb is installable ... or else give great instructions.

setup(
    name="somn",
    packages=find_packages(exclude=("__pycache__",)),
    data_files=[],
    version="0.1",
    author="N. Ian Rinehart",
    author_email="nir2@illinois.edu",
    install_requires=[
        "seaborn",
        "Pillow",
        "Pyarrow",
        "scikit-learn>=0.22.1",
        "attrs",
        ## Pip installs:
        # "tensorflow=2.12.0",
        # "keras-tuner",
        # "jupyter",
    ],
    python_requires=">=3.9",
)
