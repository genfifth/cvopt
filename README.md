# cvopt
cvopt(cross validation optimizer) is python module for machine learning's parameter search and feature selection, integrated visualization and archive log.   
This module has API like scikit-learn cross validation class and easy to use.

![readme_00](https://github.com/genfifth/cvopt/blob/master/etc/images/readme_00.PNG)

# Features
* Optimize parameters and feature selections.
* Integrated visualization and archive log.
* API like scikit-learn cross validation class.
   * Support Algorithm
      * Sequential Model Based Global Optimization (Hyperopt)
      * Bayesian Optimization (GpyOpt)
      * Genetic Algorithm

# Installation   
```bash
$ pip install git+https://github.com/genfifth/cvopt
```
requires:   
* Python3
* NumPy
* pandas
* scikit-learn
* Hyperopt
* Gpy
* GpyOpt
* bokeh
   
# Quick start -search can be written in 5 lines.-
```python
param_distributions = {"penalty": search_category(['l1', 'l2']), "C": search_numeric(0, 3, "float"), 
                       "tol" : search_numeric(0, 4, "float"),  "class_weight" : search_category([None, "balanced"])}
feature_groups = np.random.randint(0, 5, Xtrain.shape[1]) 
hpcv = SimpleoptCV(estimator=LogisticRegression(), param_distributions=param_distributions)
hpcv.fit(Xtrain, ytrain, feature_groups=feature_groups)
```
   
# Documents
[Basic usage](https://github.com/genfifth/cvopt/blob/master/notebooks/basic_usage.ipynb)
[Basic usage(jp)](https://github.com/genfifth/cvopt/blob/master/notebooks/basic_usage_jp.ipynb)
   
[API reference](https://genfifth.github.io/cvopt/)

# Changelog
[Log](https://github.com/genfifth/cvopt/blob/master/Changelog.md)   
