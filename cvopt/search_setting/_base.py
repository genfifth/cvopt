import numpy as np, scipy as sp
import types
from hyperopt import hp
from hyperopt.pyll import scope

class ParamDist(dict):
    """
    cvopt standard param setting class.
    """
    pass


def search_category(categories):
    """
    Set search target distribution for categorical variable.

    Parameters
    ----------
    categories: list
        search target categories.

    Returns
    ----------
    cvopt.search_setting.PramDist
        setting class
    """
    if not isinstance(categories, list):
        raise ValueError("categories is must be list")
    paramdist = ParamDist(valtype="category", categories=categories)
    return paramdist


def search_numeric(low, high, dtype):
    """
    Set search target distribution for numerical variable.

    Parameters
    ----------
    low: int or float
        lower limit of search range.

    high: int or float
        high limit of search range.

    dtype: "int" or "float"
        variable's dtype.

    Returns
    ----------
    cvopt.search_setting.PramDist
        setting class
    """
    if not dtype in ["int", "float"]:
        raise ValueError('dtype is must be "int" or "float"')
        
    paramdist = ParamDist(valtype="numeric", 
                          low=low, 
                          high=high, 
                          dtype=dtype, 
                         )
    return paramdist


def to_func(x):
    if isinstance(x, types.FunctionType):
        return x
    else:
        def f(_x):
            return x
        return f


class category_sampler:
    def __init__(self, categories):
        self.categories = categories
        self.dist = sp.stats.randint(low=0, high=len(categories))
        
    def rvs(self):
        return self.categories[self.dist.rvs()]

def get_params(param_distributions, tgt_key=None):
    """
    get params from param_distributions (dict, key:param_name, val:scipy.stat class).
    """
    if tgt_key is None:
        ret = {}
        for key in param_distributions.keys():
            ret[key] = param_distributions[key].rvs()
        return ret
    else:
        return {tgt_key:param_distributions[tgt_key].rvs()}


@scope.define
def hpint(low, high):
    return max(low, high)

def _conv_hyperopt_param_dist(param_name, param_dist):
    if param_dist["valtype"] == "numeric":
        if param_dist["dtype"] == "int":
            param_dist = scope.hpint(int(param_dist["low"]), hp.randint(param_name, int(param_dist["high"])))
        elif param_dist["dtype"] == "float":
            param_dist = hp.uniform(param_name, 
                                    param_dist["low"], 
                                    param_dist["high"])            
    elif param_dist["valtype"] == "category":
        param_dist = hp.choice(param_name, param_dist["categories"])

    return param_dist

def _conv_gpyopt_param_dist(param_name, param_dist):
    if param_dist["valtype"] == "numeric":
        if param_dist["dtype"] == "int":
            param_dist = {"name":param_name, "type":"discrete", 
                          "domain":np.arange(int(param_dist["low"]), int(param_dist["high"])+1).astype(int)}
        elif param_dist["dtype"] == "float":
            param_dist = {"name":param_name, "type":"continuous", 
                          "domain":(param_dist["low"], param_dist["high"])}         
    elif param_dist["valtype"] == "category":
        param_dist = {"name":param_name, "type":"categorical", 
                      "domain":np.arange(len(param_dist["categories"])), 
                      "categories":param_dist["categories"]}
    return param_dist

def _conv_ga_param_dist(param_name, param_dist):
    if param_dist["valtype"] == "numeric":
        if param_dist["dtype"] == "int":
            param_dist = sp.stats.randint(low=param_dist["low"], high=param_dist["high"])
        elif param_dist["dtype"] == "float":
            param_dist = sp.stats.uniform(loc=param_dist["low"], scale=param_dist["high"]-param_dist["low"])            
    elif param_dist["valtype"] == "category":
        param_dist = category_sampler(categories=param_dist["categories"])

    return param_dist

def conv_param_distributions(param_distributions, backend):
    """
    Convert param_distributions from cvopt style to backend style.
    """
    if backend == "hyperopt":
        ret = {}
    elif backend == "bayesopt":
        ret = []
    elif backend == "gaopt":
        ret = {}
    
    for param_name in param_distributions:
        if type(param_distributions[param_name]) == ParamDist:
            try:
                if backend == "hyperopt":
                    ret[param_name] = _conv_hyperopt_param_dist(param_name, param_distributions[param_name])
                elif backend == "bayesopt":
                    ret.append(_conv_gpyopt_param_dist(param_name, param_distributions[param_name]))
                elif backend == "gaopt":
                    ret[param_name] = _conv_ga_param_dist(param_name, param_distributions[param_name])
            except Exception as e:
                raise ValueError("parameter:"+ param_name + "'s setting is not supported.")
                
        else:
            if backend == "hyperopt":
                ret[param_name] = param_distributions[param_name]
            elif backend == "bayesopt":
                if(param_distributions[param_name]["type"]=="categorical") & ("categories" not in param_distributions[param_name]):
                    raise Exception("If type is categorical, parameter_distributions's value must have `categories` key.")
                ret.append(param_distributions[param_name])
            elif backend == "gaopt":
                if isinstance(param_distributions[param_name], sp.stats._distn_infrastructure.rv_frozen):
                    ret[param_name] = param_distributions[param_name]
                else:
                    raise Exception("parameter_distributions's value must be search_setting.search_numeric, search_setting.search_category, or scipy.stats class.")
                    
    return ret


def decode_params(params, param_distributions, backend):
    """
    Decode params from backend style to dict(key:param name, value:param value).
    """
    if backend == "hyperopt":
        return params
    elif backend == "bayesopt":
        ret = {}
        for i, param_dist in enumerate(param_distributions):
            if param_dist["type"] == "categorical":
                ret[param_dist["name"]] = param_dist["categories"][int(params[0, i])]
            elif param_dist["type"] == "discrete":
                ret[param_dist["name"]] = int(params[0, i])
            else:
                ret[param_dist["name"]] = params[0, i]
        return ret  
    elif backend == "gaopt":
        return params
