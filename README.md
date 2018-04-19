# cvopt -to simplify Data Science-
cvopt (cross validation optimizer) is python module for machine learning's parameter search and feature selection.
To simplify modeling, in cvopt, log management and visualization are integrated and the API like scikit-learn is provided.

![readme_00](https://github.com/genfifth/cvopt/blob/master/etc/images/readme_00.PNG)

In Data Science modeling, sometimes would like to ...
* Use various search algorithms on the same interface.
* Optimize parameters and feature selections simultaneously.
* Integrate log management and its visualization into search API.

To make these simpler, cvopt was created.

# Features
* API like scikit-learn.
   * Support Algorithm:
      * Sequential Model Based Global Optimization (Hyperopt)
      * Bayesian Optimization (GpyOpt)
      * Genetic Algorithm
      * Random Search
* Optimization of parameters and feature selections.
* Integration of log management and visualization.


# Installation   
```bash
$ pip install Gpy
$ pip install cvopt
```
Requires:   
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
opt = SimpleoptCV(estimator=LogisticRegression(), param_distributions=param_distributions)
opt.fit(Xtrain, ytrain, feature_groups=feature_groups)
```
   
# Documents
Basic usage[(en)](https://github.com/genfifth/cvopt/blob/master/notebooks/basic_usage.ipynb)/[(jp)](https://github.com/genfifth/cvopt/blob/master/notebooks/basic_usage_jp.ipynb)
   
[Keras sample](https://github.com/genfifth/cvopt/blob/master/notebooks/keras_sample.ipynb)
   
[API Reference](https://genfifth.github.io/cvopt/)

# Changelog
[Log](https://github.com/genfifth/cvopt/blob/master/Changelog.md)   
