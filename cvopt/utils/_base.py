import os, copy, warnings, scipy, types
import pandas as pd, numpy as np
from sklearn.base import clone
from sklearn.externals.joblib import dump, load

def mk_dir(path, error_level=0, msg=None):
    """
    Make directory.    
    """
    if msg is None:
        msg = ""
    
    if os.path.isdir(path):
        if error_level==1:
            warnings.warn("%s Could not be made (already exist) " %path+msg)
        elif error_level==2:
            raise Exception("%s Could not be made (already exist) " %path+msg)
    else:
        os.makedirs(path, mode=0o777)
        os.chmod(path, mode=0o777)


def to_nparray(Xy, ravel_1d):
    """
    Convert to np.array

    Parameters
    ----------
    Xy
        X or y

    ravel_1d: bool
        When True, Xy.shape=(n,1) -> Xy.shape=(n,)
    """
    if isinstance(Xy, (pd.core.frame.DataFrame, pd.core.frame.Series)):
        Xy = Xy.values

    if (ravel_1d) and (len(Xy.shape) == 2) and (Xy.shape[1] == 1):
        return Xy.ravel()
    else:
        return Xy


def chk_Xy(Xy, none_error, ravel_1d, msg_sjt):
    """
    Check class of X or y.
    """
    base_msg = " must be numpy.array, pandas.DataFrame or scipy.sparse."
    if Xy is None:
        if none_error:
            raise TypeError(msg_sjt+base_msg)
    elif scipy.sparse.issparse(Xy):
    	return Xy
    elif isinstance(Xy, (np.ndarray, pd.core.frame.DataFrame)):
    	return to_nparray(Xy, ravel_1d)

    raise TypeError(msg_sjt+base_msg)

def compress(condition, a, axis):
    if isinstance(a, np.ndarray):
        return np.compress(condition=condition, a=a, axis=axis)
    elif scipy.sparse.issparse(a):
        if axis == 0:
            return a[condition]
        elif axis == 1:
            return a[:, condition]
        elif axis == 2:
            return a[:, :, condition]
        elif axis == 3:
            return a[:, :, :, condition]
        elif axis == 4:
            return a[:, :, :, :, condition]
        else:
            raise ValueError("axis must be 0-4")
    else:
        raise ValueError("input array must be numpy.array or scipy.sparse")


def make_cloner(cloner):
    if isinstance(cloner, types.FunctionType):
        def clone_estimator(estimator, params):
            estimator = cloner(estimator).set_params(**params)
            return estimator
    elif cloner == "sklearn":
        def clone_estimator(estimator, params):
            try:
                estimator = clone(estimator).set_params(**params)
            except RuntimeError:
                estimator = copy.deepcopy(estimator).set_params(**params)
            return estimator
    else:
        raise Exception("cloner=" + str(cloner)+" is not supported.")
    return clone_estimator

def make_saver(saver):
    if isinstance(saver, types.FunctionType):
        return saver
    elif saver == "sklearn":
        def saver(estimator, path):
            dump(estimator, path+".pkl")
    else:
        raise Exception("saver=" + str(saver)+" is not supported.")
    return saver
    
def make_loader(loader):
    if isinstance(loader, types.FunctionType):
        return loader
    elif loader == "sklearn":
        def loader(path):
            return load(path+".pkl")
    else:
        raise Exception("loader=" + str(loader)+" is not supported.")
    return loader

def to_label(y):
    if (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
        return y
    else:
        # Supposing y is 1 hot encoding, convert to label.
        return np.argmax(y, axis=1)


def scale(val, from_range, to_range):
    tmp = (val - from_range[0]) / (from_range[1] - from_range[0])
    return tmp * (to_range[1]- to_range[0]) + to_range[0]
            
    
        