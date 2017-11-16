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
   
# Quick start -search can be written in 5 lines.-
```python
param_distributions = {"penalty":hp.choice("penalty", ["l1", "l2"]), "tol":hp.loguniform("tol", -4, -2), 
                       "C":hp.loguniform("C", -3, 3), "class_weight":hp.choice("class_weight", [None, "balanced"])}
feature_groups = np.random.randint(0, 5, Xtrain.shape[1]) 
hpcv = hyperoptCV(estimator=LogisticRegression(), param_distributions=param_distributions)
hpcv.fit(Xtrain, ytrain, feature_groups=feature_groups)
```
   
# Document
[Basic usage](https://github.com/genfifth/cvopt/blob/master/docs/basic_usage.ipynb)
API reference