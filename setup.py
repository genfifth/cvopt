#!usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages

setup(
        name             = "cvopt",
        version          = "0.3.3",
        description      = "Parameter search and feature selection's class, Integrated visualization and archive log.",
        license          = "BSD-2-Clause",
        author           = "gen/5",
        author_email     = "gen_fifth@outlook.jp",
        url              = "https://github.com/genfifth/cvopt.git",
        packages         = find_packages(),
        install_requires = ["numpy>=1.14", 
                            "pandas>=0.22.0", 
                            "scikit-learn>=0.19.1", 
                            "hyperopt>=0.1", 
                            "networkx==1.11", 
                            "GPy>=1.9.2", 
                            "gpyopt>=1.2.1", 
                            "bokeh>=0.12.14"],
        )
