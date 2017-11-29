import os, copy, warnings
import pandas as pd, numpy as np
from sklearn.base import clone


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

    # Arguments
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
    base_msg = " must be np.array or pd.DataFrame."
    if Xy is None:
        if none_error:
            raise TypeError(msg_sjt+base_msg)
    elif not isinstance(Xy, (np.ndarray, pd.core.frame.DataFrame)):
        raise TypeError(msg_sjt+base_msg)

    return to_nparray(Xy, ravel_1d)


def clone_estimator(estimator, params):
    """
    Clone estimator and set params.
    """
    try:
        estimator = clone(estimator).set_params(**params)
    except RuntimeError:
        estimator = copy.deepcopy(estimator).set_params(**params)
    return estimator
