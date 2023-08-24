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
        "seaborn==0.12.2",
        "pandas==1.5.2",
        "Pillow==9.3.0",
        "pyarrow==8.0.0",
        "numpy>=1.19.0",
        "PyYAML>=5.3",
        "scikit-learn>=0.22.1",
        "scipy>=1.4.1",
        "colorama>=0.4.4",
        # "crest==2.12",
        "attrs==23.1.0",
        "tensorflow==2.12.0",
        "keras-tuner==1.3.5",
        "rdkit==2023.03.1",
    ],
    python_requires=">=3.10",
)
