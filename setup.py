#!usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages

setup(
        name             = "cvopt",
        version          = "0.4.3",
        description      = "Parameter search and feature selection's class, Integrated visualization and archive log.",
        license          = "BSD-2-Clause",
        author           = "gen/5",
        author_email     = "gen_fifth@outlook.jp",
        url              = "https://github.com/genfifth/cvopt.git",
        packages         = find_packages(),
        install_requires = ["numpy>=1.17.2", 
                            "pandas>=0.25.1", 
                            "scikit-learn>=0.22", 
                            "hyperopt>=0.2.2", 
                            "GPy>=1.9.9", 
                            "gpyopt>=1.2.5", 
                            "bokeh>=1.3.4"],
        )
