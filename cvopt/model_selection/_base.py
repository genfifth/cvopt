import os, sys, copy, time
import pandas as pd, numpy as np
from datetime import datetime
from abc import ABCMeta, abstractmethod

from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import BaseEstimator, is_classifier
from sklearn.metrics import SCORERS, make_scorer
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import check_cv

from ..model_selection import _setting as st
from ..utils._base import chk_Xy, clone_estimator
from ..utils._logger import CVSummarizer


class BaseSearcher(BaseEstimator, metaclass=ABCMeta):
    """
    Base class of cross validation optimizer.

    # Class variables
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


    @abstractmethod
    def _mk_feature_select_patam_distribution(self, key, distribution):
        pass

    @abstractmethod
    def fit(self):
        pass

    def __init__(self, estimator, param_distributions, 
                 scoring, cv, n_jobs, pre_dispatch, 
                 verbose, logdir, save_estimator, model_id, refit):
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

        param_distributions = {}
        if feature_groups is None:
            self._feature_select = False
        else:
            self._feature_select = True
            feature_select_param = mk_feature_select_params(feature_groups, n_samples=len(X), 
                                                            n_features=X.shape[BaseSearcher.feature_axis])
            for key in feature_select_param.keys():
                param_distributions[key] = self._mk_feature_select_patam_distribution(key, feature_select_param[key])
        param_distributions.update(self.param_distributions)

        self._cvs = CVSummarizer(paraname_list=param_distributions.keys(), cvsize=self.n_splits_, 
                                 score_summarizer=BaseSearcher.score_summarizer, score_summarizer_name=BaseSearcher.score_summarizer_name, 
                                 valid=valid, sign=self.sign, model_id=self.model_id, verbose=self.verbose, 
                                 save_estimator=self.save_estimator, logdir=self.logdir)
        self.cv_results_ = self._cvs()

        return X, y, Xvalid, yvalid, cv, param_distributions


    def _postproc_fit(self, X, y, feature_groups):
        if self.refit:
            if self._feature_select:
                self.feature_select_ind, _, estimator_params = mk_feature_select_index(self.best_params_, feature_groups, verbose=1)
                self.best_estimator_ = clone_estimator(self.estimator, estimator_params)
            else:
                self.feature_select_ind = np.array([True]*X.shape[BaseSearcher.feature_axis])
                self.best_estimator_ = clone_estimator(self.estimator, self.best_params_)
            if y is None:
                self.best_estimator_.fit(np.compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis))
            else:
                self.best_estimator_.fit(np.compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis), y)
        if self.verbose == 1:
            sys.stdout.write("\n\rBest_score(finished):%s" %np.round(self.best_score_, 2))


    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def predict(self, X):
        X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.predict(np.compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def predict_proba(self, X):
        X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.predict_proba(np.compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def predict_log_proba(self, X):
        X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.predict_log_proba(np.compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def decision_function(self, X):
        X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.decision_function(np.compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def transform(self, X):
        X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.transform(np.compress(self.feature_select_ind, X, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def inverse_transform(self, Xt):
        Xt = chk_Xy(Xt, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.inverse_transform(np.compress(self.feature_select_ind, Xt, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def apply(self, X):
        X = chk_Xy(X, none_error=True, ravel_1d=False, msg_sjt="X")
        return self.best_estimator_.apply(np.compress(self.feature_select_ind, Xt, axis=BaseSearcher.feature_axis))

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def classes_(self):
        return self.best_estimator_.classes_



def mk_feature_select_params(feature_groups, n_samples, n_features):
    """
    Set param_distributions for feature_select
    """
    if len(feature_groups) != n_features:
        raise Exception("feature_groups length must be equal to X length(n_sample)")
    
    feature_group_names = np.unique(feature_groups)
    feature_group_names.sort()
    
    ret = {}
    for i in feature_group_names:
        tmp = st.FEATURE_SELECT_PARAMNAME_PREFIX + str(i)
        if i == st.ALWAYS_USED_FEATURE_GROUP_ID:
            ret[tmp] = [True]
        elif n_samples == 1:
            ret[tmp] = [True]
        else:
            ret[tmp] = [True, False]
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

        # Arguments
        test_data: tuple(X, y)
            When test_data is not None, ignore test_ind and use test_data in compute score_test.
        """
        if train_ind is None:
            train_ind = np.arange(len(X))

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

            

