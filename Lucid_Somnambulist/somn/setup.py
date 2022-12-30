"""
    This module enables `pip install ./`
    Do not run this script standalone.
    Refer to package readme for further details.
"""

from setuptools import setup, find_packages
from glob import glob
import os

### Add dependencies later; this is not done as of 9-9-2022

setup(
    name="molli",
    packages=find_packages(exclude=("__pycache__", )),
    data_files=[],
    version="0.1",
    author="N. Ian Rinehart",
    author_email="nir2@illinois.edu",
    install_requires=[
        "seaborn",
        "pandas",
        "Pillow",
        "pyarrow",
        "numpy>=1.19.0",
        "PyYAML>=5.3",
        "scikit-learn>=0.22.1",
        "scipy>=1.4.1",
        "colorama>=0.4.4",
    ],
    python_requires=">=3.9",
)
