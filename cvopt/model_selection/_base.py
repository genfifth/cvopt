import os, sys, copy, time
import pandas as pd, numpy as np
from datetime import datetime
from abc import ABCMeta, abstractmethod

from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import BaseEstimator, is_classifier
from sklearn.metrics import SCORERS, make_scorer
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import check_cv
from sklearn.externals.joblib import Parallel, delayed
from hyperopt import STATUS_OK, STATUS_FAIL

from ..model_selection import _setting as st
from ..search_setting._base import conv_param_distributions, search_category, decode_params
from ..utils._base import chk_Xy, clone_estimator, compress, make_saver, to_label
from ..utils._logger import CVSummarizer

class BaseSearcher(BaseEstimator, metaclass=ABCMeta):
    """
    Base class of cross validation optimizer.

    Class Variables
    ---------------
    score_summarizer: function(scores per cv)
        Score summarize function.
    score_summarizer_name: str
        This is used in logfile.
    feature_axis: int
        feature_select target axis. 
    """
    score_summarizer = np.mean
    score_summarizer_name = "mean"
    feature_axis = 1

    def __init__(self, estimator, param_distributions, 
                 scoring, cv, n_jobs, pre_dispatch, 
                 verbose, logdir, save_estimator, saver, model_id, refit, backend):
        # BACKLOG: Implement iid option(sklearn GridSearchCV have this option).
        
        self.estimator = estimator
        self.scoring = check_scoring(estimator, scoring=scoring)
        if hasattr(self.scoring, "_sign"):
            self.sign = self.scoring._sign
        else:
            self.sign = 1
            # In this case, scoring is None & use estimator default scorer.
            # Because scikit-learn default scorer is r2_score or accuracy, score greater is better.
            # So sign = 1.

        self.cv = cv
        self.param_distributions = copy.deepcopy(param_distributions)
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.logdir = logdir
        self.save_estimator = save_estimator
        self.saver = saver
        self.backend = backend

        if model_id is None:
            self.model_id = datetime.today().strftime("%Y%m%d_%H%M%S")
        else:
            self.model_id = model_id
        self.refit = refit


    def _preproc_fit(self, X, y, validation_data, feature_groups):
        X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
        y = chk_Xy(y, none_error=False, ravel_1d=True, msg_sjt="y")
        if validation_data is None:
            Xvalid, yvalid = None, None
            valid = False
        else:
            Xvalid = chk_Xy(validation_data[0], none_error=False, ravel_1d=False, msg_sjt="Xvalid")
            yvalid = chk_Xy(validation_data[1], none_error=False, ravel_1d=True, msg_sjt="yvalid")
            valid = True
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        self.n_splits_ = cv.get_n_splits()

        if feature_groups is None:
            self._feature_select = False
            param_distributions = {}
        else:
            self._feature_select = True
            param_distributions = mk_feature_select_params(feature_groups, n_samples=X.shape[0], 
                                                           n_features=X.shape[BaseSearcher.feature_axis])
        param_distributions.update(self.param_distributions)
        
        self._cvs = CVSummarizer(paraname_list=param_distributions.keys(), cvsize=self.n_splits_, 
                                 score_summarizer=BaseSearcher.score_summarizer, score_summarizer_name=BaseSearcher.score_summarizer_name, 
                                 valid=valid, sign=self.sign, model_id=self.model_id, verbose=self.verbose, 
                                 save_estimator=self.save_estimator, logdir=self.logdir)
        self.cv_results_ = self._cvs()

        return X, y, Xvalid, yvalid, cv, conv_param_distributions(param_distributions, backend=self.backend)


    def _postproc_fit(self, X, y, feature_groups, best_params, best_score):
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        if self.refit:
            if self._feature_select:
                self.feature_select_ind, _, estimator_params = mk_feature_select_index(self.best_params_, feature_groups, verbose=1)
                self.best_estimator_ = clone_estimator(self.estimator, estimator_params)
            else:
                self.feature_select_ind = np.array([True]*X.shape[BaseSearcher.feature_axis])
                self.best_estimator_ = clone_estimator(self.estimator, self.best_params_)
            if y is None:
                self.best_estimator_.fit(compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis))
            else:
                self.best_estimator_.fit(compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis), y)
        if self.verbose == 1:
            sys.stdout.write("\n\rBest_score(finished):%s" %np.round(self.best_score_, 2))

    def _random_scoring(self, y):
        score = []
        for i in range(10):
            score.append(self.scoring._sign*self.scoring._score_func(y, np.random.permutation(y)))
        return np.median(score)

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def predict(self, X):
        """
        Call predict on the estimator with the best found parameters.

        Parameters
        ----------       
        X :numpy.array, pandas.DataFrame or scipy.sparse, shape(axis=0) = (n_samples)
            Features. Detail depends on estimator.
        """
        X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.predict(compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def predict_proba(self, X):
        """
        Call predict_proba on the estimator with the best found parameters.
        
        Parameters
        ----------       
        X :numpy.array, pandas.DataFrame or scipy.sparse, shape(axis=0) = (n_samples)
            Features. Detail depends on estimator.
        """
        X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.predict_proba(compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def predict_log_proba(self, X):
        """
        Call predict_log_proba on the estimator with the best found parameters.
        
        Parameters
        ----------       
        X :numpy.array, pandas.DataFrame or scipy.sparse, shape(axis=0) = (n_samples)
            Features. Detail depends on estimator.
        """
        X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.predict_log_proba(compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def decision_function(self, X):
        """
        Call decision_function on the estimator with the best found parameters.
        
        Parameters
        ----------       
        X :numpy.array, pandas.DataFrame or scipy.sparse, shape(axis=0) = (n_samples)
            Features. Detail depends on estimator.
        """
        X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.decision_function(compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def transform(self, X):
        """
        Call transform on the estimator with the best found parameters.
        
        Parameters
        ----------       
        X :numpy.array, pandas.DataFrame or scipy.sparse, shape(axis=0) = (n_samples)
            Features. Detail depends on estimator.
        """
        X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.transform(compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def inverse_transform(self, Xt):
        """
        Call inverse_transform on the estimator with the best found parameters.
        
        Parameters
        ----------       
        Xt :numpy.array, pandas.DataFrame or scipy.sparse, shape(axis=0) = (n_samples)
            Features. Detail depends on estimator.
        """
        Xt = chk_Xy(Xt, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.inverse_transform(compress(self.feature_select_ind, Xt, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def apply(self, X):
        """
        Call apply on the estimator with the best found parameters.
        
        Parameters
        ----------       
        X :numpy.array, pandas.DataFrame or scipy.sparse, shape(axis=0) = (n_samples)
            Features. Detail depends on estimator.
        """
        X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.apply(compress(self.feature_select_ind, Xt, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def classes_(self):
        """
        Retern list of target classes.
        """
        return self.best_estimator_.classes_



def mk_feature_select_params(feature_groups, n_samples, n_features):
    """
    Set param_distributions for feature_select.
    """
    if len(feature_groups) != n_features:
        raise Exception("feature_groups length must be equal to X length(n_sample)")
    
    feature_group_names = np.unique(feature_groups)
    feature_group_names.sort()
    
    ret = {}
    for i in feature_group_names:
        tmp = st.FEATURE_SELECT_PARAMNAME_PREFIX + str(i)
        if i == st.ALWAYS_USED_FEATURE_GROUP_ID:
            ret[tmp] = search_category([True])
        elif n_samples == 1:
            ret[tmp] = search_category([True])
        else:
            ret[tmp] = search_category([True, False])
    return ret



def mk_feature_select_index(params, feature_groups, verbose):
    """
    Make feature select index.
    """
    true_group_names = []
    other_params = copy.deepcopy(params)
    feature_params = {}
    for key in params.keys():
        if st.FEATURE_SELECT_PARAMNAME_PREFIX in key:
            feature_params[key] = other_params.pop(key)
            if params[key]:
                true_group_names.append(int(key.split(st.FEATURE_SELECT_PARAMNAME_PREFIX)[-1]))

    if verbose:
        return np.in1d(feature_groups, true_group_names), feature_params, other_params
    else:
        return np.in1d(feature_groups, true_group_names)



def fit_and_score(estimator, X, y, scoring, train_ind=None, test_ind=None, test_data=None):
        """
        Run fit and compute evaluation index.

        Parameters
        ---------- 
        test_data: tuple(X, y)
            When test_data is not None, ignore test_ind and use test_data in compute score_test.
        """
        if train_ind is None:
            train_ind = np.arange(X.shape[0])

        start = time.time()
        estimator.fit(X[train_ind], y[train_ind])
        fittime = time.time() - start

        start = time.time()
        score_train = scoring(estimator, X[train_ind], y[train_ind])
        scoretime = time.time() - start
        
        if test_data is None:
            score_test = scoring(estimator, X[test_ind], y[test_ind])
        else:
            score_test = scoring(estimator, *test_data)

        return score_train, score_test, fittime, scoretime, estimator



def _obj_return(score, succeed, backend):
    if backend == "hyperopt":
        if succeed:
            return {"loss":-1.0*score, "status":STATUS_OK}
        else:
            return {"loss":score, "status":STATUS_FAIL}
    
    elif (backend == "bayesopt") or (backend == "gaopt"):
        return -1.0*score



def mk_objfunc(X, y, groups, feature_groups, feature_axis, estimator, scoring, cv, 
               param_distributions, backend, failedscore, saver, 
               score_summarizer=np.mean, 
               Xvalid=None, yvalid=None, n_jobs=1, pre_dispatch="2*n_jobs", 
               cvsummarizer=None, save_estimator=0, min_n_features=2):
    """
    Function to make search objective function(input:params, output:evaluation index)
    """
    n_splits_ = cv.get_n_splits()
    cvs = cvsummarizer
    saver = make_saver(saver)

    def obj(params):
        start_time = datetime.now()
        params = decode_params(params=params, param_distributions=param_distributions, backend=backend)

        cvs.display_status(params=params, start_time=start_time)

        if feature_groups is None:
            feature_select = False
            feature_select_ind = np.array([True]*X.shape[feature_axis])
            estimator_params = copy.deepcopy(params)
        else:
            feature_select = True
            feature_select_ind, _, estimator_params = mk_feature_select_index(params, feature_groups, verbose=1)
            if feature_select_ind.sum() < min_n_features:
                score = np.nan
                if cvs is not None:
                    end_time = datetime.now()
                    cvs.store_cv_result(cv_train_scores=[np.nan]*n_splits_, cv_test_scores=[np.nan]*n_splits_, params=params, 
                                        fit_times=[np.nan]*n_splits_, score_times=[np.nan]*n_splits_, feature_select=feature_select,
                                        X_shape=compress(feature_select_ind, X, axis=feature_axis).shape, 
                                        start_time=start_time, end_time=end_time, 
                                        train_score=np.nan, validation_score=np.nan)
                return _obj_return(score=failedscore, succeed=False, backend=backend)

        # cross validation
        cv_train_scores = []
        cv_test_scores = []
        fit_times = []
        score_times = []

        ret_p = Parallel(
            n_jobs=n_jobs, pre_dispatch=pre_dispatch, 
        )(delayed(fit_and_score)(estimator=clone_estimator(estimator, estimator_params), 
                                 X=compress(feature_select_ind, X, axis=feature_axis),  
                                 y=y, 
                                 train_ind=train_ind, test_ind=test_ind, 
                                 scoring=scoring)
        for train_ind, test_ind in cv.split(compress(feature_select_ind, X, axis=feature_axis), to_label(y), groups))

        # evaluate validation data
        if Xvalid is None:
            train_score, validation_score = np.nan, np.nan
        else:
            train_score, validation_score, _, _, estimator_test = fit_and_score(
                estimator=clone_estimator(estimator, estimator_params), 
                X=compress(feature_select_ind, X, axis=feature_axis),  
                y=y, 
                test_data=(compress(feature_select_ind, Xvalid, axis=feature_axis), yvalid), 
                scoring=scoring)

        # summarize
        if (cvs.logdir is not None) & (save_estimator > 0):
            estimators_cv = []
            path = os.path.join(cvs.logdir, "estimators", cvs.model_id)
            name_prefix = cvs.model_id + "_index" + "{0:05d}".format(len(cvs.cv_results_["params"]))
            for cnt, (i, j, k, l, m) in enumerate(ret_p):
                cv_train_scores.append(i)
                cv_test_scores.append(j)
                fit_times.append(k)
                score_times.append(l)

                saver(m, os.path.join(path, name_prefix+"_split"+"{0:02d}".format(cnt)))

            if (save_estimator > 1):
                if Xvalid is None:
                    estimator_test = clone_estimator(estimator, estimator_params)
                    estimator_test.fit(compress(feature_select_ind, X, axis=feature_axis), y)
                saver(estimator_test, os.path.join(path, name_prefix+"_test"))
        else:
            for i, j, k, l, m in ret_p:
                cv_train_scores.append(i)
                cv_test_scores.append(j)
                fit_times.append(k)
                score_times.append(l)

        if cvs is not None:
            end_time = datetime.now()
            cvs.store_cv_result(cv_train_scores=cv_train_scores, cv_test_scores=cv_test_scores, params=params, 
                                fit_times=fit_times, score_times=score_times, feature_select=feature_select,
                                X_shape=compress(feature_select_ind, X, axis=feature_axis).shape,
                                start_time=start_time, end_time=end_time, 
                                train_score=train_score, validation_score=validation_score)
        score = score_summarizer(cv_test_scores)
        return _obj_return(score=score, succeed=True, backend=backend)
    return obj