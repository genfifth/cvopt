# cvopt
cvopt(cross validation optimizer) is python module for machine learning's parameter search and feature selection, integrated visualization and archive log.   
This module has API like scikit-learn cross validation class and easy to use.

![readme_00](https://github.com/genfifth/images/blob/master/cvopt/readme_00.PNG)

# Features
* Optimize parameters and feature selections.
* Integrated visualization and archive log.
* API like scikit-learn cross validation class.

# Installation   
```bash
$ pip install git+https://github.com/genfifth/cvopt
```
requires:   
* Python3
* NumPy
* pandas
* scikit-learn
* hyperopt
* bokeh
   
# Quick start -Search can be written in 5 lines.-
```python
from cvopt import hyperoptCV
param_distributions = {"penalty": hp.choice("penalty", ["l1", "l2"]), "C": hp.choice("C", [1e-1, 1e-0, 1e1]),
                       "class_weight" : hp.choice("class_weight", [None, "balanced"]),}
hpcv = hyperoptCV(estimator=LogisticRegression(), param_distributions=param_distributions)
hpcv.fit(Xtrain, ytrain)
```
   
# Usage
Please see [example](https://github.com/genfifth/cvopt/blob/master/example/example.ipynb)
