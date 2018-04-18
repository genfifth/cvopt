import os, warnings
import pandas as pd, numpy as np

from ._base import chk_Xy, make_loader, to_label
from ..model_selection import _setting as st


def extract_params(logdir, model_id, target_index, feature_groups=None):
    """
    Extract parameters from cvopt logfile.
    
    Parameters
    ----------
    logdir: str.
        cvopt's log directory path.
    
    model_id: str.
        cvopt's model id.
        
    target_index: int.
        Logfile index(start:0). 
        Parameters correspond to index is extracted.
        
    feature_groups: array-like, shape = (n_samples,) or None, default=None.
        cvopt feature_groups.
        When feature_groups is None,  feature_param and feature_select_flag in returns is None.
        
        feature select flag is bool vector. 
        If this value is True, optimizer recommend using corresponding column.
    
    Returns
    -------  
    estimator_params: dict
        estimator parameters of the target model.
    
    feature_params: dict or None
        feature parameters of the target model.
    
    feature_select_flag: numpy array or None
        feature select flag of the target model.
    
    """
    logfile = pd.read_csv(os.path.join(logdir, "cv_results", model_id+".csv"))
    logfile.set_index("index", inplace=True)
    if not logfile.index.is_unique:
        raise Exception("%s index must be unique" %os.path.join(logdir, "cv_results", model_id+".csv"))

    params = eval(logfile.loc[target_index, "params"])
    estimator_params = dict()
    feature_params = dict()
    feature_select_flag = np.array(feature_groups).astype(str)
    for key in params.keys():
        if st.FEATURE_SELECT_PARAMNAME_PREFIX in key:
            feature_params[key] = params[key]
            feature_group_id = key.split(st.FEATURE_SELECT_PARAMNAME_PREFIX)[-1]
            feature_select_flag = np.where(feature_select_flag==feature_group_id, feature_params[key], feature_select_flag)
        else:
            estimator_params[key] = params[key]
        
    if feature_groups is None:
        if len(feature_params) > 0:
            warnings.warn("This log file include feature select setting. Please specify feature_groups if necessary.")
        return estimator_params, None, None
    else:
        if len(feature_params) == 0:
            warnings.warn("This log file don't include feature select setting. So return is (estimator_params, None, None)")
            return estimator_params, None, None
        feature_select_flag = np.where(feature_select_flag=="True", True, False) 
        return estimator_params, feature_params, feature_select_flag
    

def mk_metafeature(X, y, logdir, model_id, target_index, cv, 
                   validation_data=None, feature_groups=None, estimator_method="predict", merge=True, loader="sklearn"):
    """
    Make meta feature for stacking(https://mlwave.com/kaggle-ensembling-guide/)
    
    Parameters
    ----------
    X :np.ndarray or pd.core.frame.DataFrame, shape(axis=0) = (n_samples)
        Features that was used in optimizer training. Detail depends on estimator.
        Meta feature correspond to X is made using cross validation's estimator.
        
    y: np.ndarray or pd.core.frame.DataFrame, shape(axis=0) = (n_samples) or None, default=None.
        Target variable that was used in optimizer training. Detail depends on estimator.
        
    logdir: str.
        cvopt's log directory path.
    
    model_id: str.
        cvopt's model id.
        
    target_index: int.
        Logfile index(start:0). 
        The estimator correspond to index is used to make meta feature.
        
    cv: scikit-learn cross-validator
        Cross validation setting that was used in optimizer training.
        
    validation_data: tuple(X, y) or None, default=None.
        Detail depends on estimator.
        Meta feature correspond to validation_data is made using the estimator 
        which is fitted whole train data.
        
    feature_groups: array-like, shape = (n_samples,) or None, default=None.
        cvopt feature_groups that was used in optimizer training.
        
    estimator_method: str, default="predict".
        Using estimator's method to make meta feature.
    
    merge: bool, default=True.
        if True, return matrix which result per cv is merged into.

    loader: str or function, default="sklearn".
        estimator`s loader.

        * `sklearn`: use `sklearn.externals.joblib.load`. Basically for scikit-learn.

        * function: function whose variable is estimator`s path.
    
    Returns
    -------
    X_meta or X_meta, X_meta_validation_data: np.ndarray or tuple of np.ndarray.
        When validation_data is input, return tuple.
        
    """
    loader = make_loader(loader)
    X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
    y = chk_Xy(y, none_error=False, ravel_1d=True, msg_sjt="y")
    if validation_data is not None:
        Xvalid = chk_Xy(validation_data[0], none_error=False, ravel_1d=False, msg_sjt="Xvalid")
        yvalid = chk_Xy(validation_data[1], none_error=False, ravel_1d=True, msg_sjt="yvalid")        
        
    if feature_groups is not None:
        _, _, feature_select_flag  = extract_params(logdir=logdir, model_id=model_id, 
                                                    target_index=target_index, feature_groups=feature_groups)
        X = X[:, feature_select_flag]
        if validation_data is not None:
            Xvalid = Xvalid[:, feature_select_flag]

            
    X_meta = []
    X_ind = []
    estdir = os.path.join(logdir, "estimators", model_id)
    name_prefix = model_id + "_index" + "{0:05d}".format(target_index)
    
    #estimator = loader(os.path.join(estdir, name_prefix+"_split"+"{0:02d}".format(0)))
    #cv = check_cv(cv, y, classifier=is_classifier(estimator))
    
    for i, (_, test_index) in enumerate(cv.split(X, to_label(y))):
        estimator = loader(os.path.join(estdir, name_prefix+"_split"+"{0:02d}".format(i)))
        X_meta.append(getattr(estimator, estimator_method)(X[test_index]))
        X_ind.append(test_index)
    
    if merge:
        X_meta = np.concatenate(X_meta, axis=0)
        X_ind = np.concatenate(X_ind, axis=0)
        X_meta = X_meta[np.argsort(X_ind)]
    
    if validation_data is None:
        return X_meta
    else:
        estimator = loader(os.path.join(estdir, name_prefix+"_test"))
        return X_meta, getattr(estimator, estimator_method)(Xvalid)
