#!usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages

setup(
        name             = "cvopt",
        version          = "0.1.0",
        description      = "Parameter search and feature selection's class, Integrated visualization and archive log.",
        license          = "BSD-2-Clause",
        author           = "gen/5",
        author_email     = "gen__@outlook.jp ",
        url              = "https://github.com/genfifth/cvopt.git",
        packages         = find_packages(),
        install_requires = ["numpy>=1.13.3", 
                            "pandas>=0.20.3", 
                            "scikit-learn>=0.19.1", 
                            "hyperopt>=0.1"
                            "bokeh>=0.12.10"],
        )