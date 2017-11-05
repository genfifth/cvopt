import os, copy
import numpy as np
from datetime import datetime
from sklearn.externals.joblib import dump, Parallel, delayed

from hyperopt import fmin, Trials, tpe, hp, space_eval, STATUS_OK, STATUS_FAIL

from .base import BaseSearcher, fit_and_score, mk_feature_select_index
from ..utils.base import mk_dir, clone_estimator
from ..utils.logger import CVSummarizer, NoteBookVisualizer

class hyperoptCV(BaseSearcher):
    def __init__(self, estimator, param_distributions, 
                 scoring=None, cv=5, 
                 algo=tpe.suggest, max_evals=10, rstate=None, 
                 n_jobs=1, pre_dispatch="2*n_jobs", 
                 verbose=0, logdir=None, save_estimator=False, model_id=None, refit=True):
        """
        # input
        estimator
            Estimator like sklearn.
        param_distributions
            Param_distributions for hyperopt.
        scoring: string or sklearn.metrics.make_scorer
            Objective of search.
        cv: sklearn_cv class or int
            Cross validation setting.
        algo
            algorithm in hyperopt.
        max_evals
            Number of search.
        rstate
            seed in hyperopt.
        n_jobs
            Number of jobs to run in parallel.
        pre_dispatch
            Controls the number of jobs that get dispatched during parallel.
        verbose
            0: don't display status
            1: display status by stdout
            2: display status by graph
        logdir
           If this path is specified, save the log.
            logdir
            |-cv_results
            | |-{model_id}.csv: search log
            | ...
            |-estimators_{model_id}
              |-{model_id}_index{search count}_split{fold count}.pkl: estimator fitted train fold data.
              ...
              |-{model_id}_index{search count}_test.pkl: estimator fitted test data(When BaseSearcher.fit.Xvalid isn't None, this is saved.).
        save_estimator
            save_estimator flag
        model_id
            model_id is used estimator's dir and filename in save.
        refit
            Refit an estimator using the best found parameters on the whole dataset.
        """
        # BACKLOG: Implement iid option(sklearn GridSearchCV have this option).

        super().__init__(estimator=estimator, param_distributions=param_distributions, 
                         scoring=scoring, cv=cv,  n_jobs=n_jobs, pre_dispatch=pre_dispatch, 
                         verbose=verbose, logdir=logdir, save_estimator=save_estimator, model_id=model_id, refit=refit)

        self.algo = algo
        self.max_evals = max_evals
        if rstate is None:
            self.rstate = rstate
        else:
            self.rstate = np.random.RandomState(int(rstate))


    def _mk_feature_select_patam_distribution(self, key, distribution):
        return hp.choice(key, distribution)
        

    def fit(self, X, y=None, validation_data=None, groups=None, 
            feature_groups=None, min_n_features=2):
        """
        X :np.ndarray or pd.core.frame.DataFrame, shape(axis=0) = (n_samples)
            Features. detail depends on estimator.

        y: np.ndarray or pd.core.frame.DataFrame, shape(axis=0) = (n_samples)
            Target variable. detail depends on estimator.

        validation_data: tuple(X, y)
            data to compute validation score. detail depends on estimator.

        groups: array-like, shape = (n_samples,) 
            Group labels for the samples used while splitting the dataset into train/test set.
            (sklearn cv's input)

        feature groups: array-like, shape = (n_samples,) 
            Group labels for the features used while fearture select.

        min_n_features: int
            When number of X's feature cols is less than min_n_features, return search failure.
            e.g. if estimator has columns sampling function, use this option to avoid X become too small and error.
        """
        X, y, Xvalid, yvalid, cv, param_distributions = self._preproc_fit(X=X, y=y, validation_data=validation_data, feature_groups=feature_groups)


        obj = mk_objfunc(return_succeed='{"loss":-1.0*score, "status":STATUS_OK}', # hyperopt minimize objective function.
                         return_failed='{"loss":None, "status":STATUS_FAIL}', 
                         X=X, y=y, groups=groups, feature_groups=feature_groups, feature_axis=BaseSearcher.feature_axis, 
                         estimator=self.estimator, scoring=self.scoring, cv=cv, 
                         score_summarizer=BaseSearcher.score_summarizer, 
                         Xvalid=Xvalid, yvalid=yvalid, n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch, 
                         cvsummarizer=self._cvs, save_estimator=self.save_estimator, min_n_features=min_n_features)

        self.trials = Trials()
        try :
            best_params = fmin(obj, param_distributions, algo=self.algo, max_evals=self.max_evals,
                               rstate=self.rstate, trials=self.trials)
        except KeyboardInterrupt:
            best_params = self.trials.argmin
        finally:
            self.best_params_ = space_eval(param_distributions, best_params)
            self.best_score_ = -1*self.trials.best_trial["result"]["loss"]

        self._postproc_fit(X=X, y=y, feature_groups=feature_groups)
        return self



def mk_objfunc(return_succeed, return_failed, 
               X, y, groups, feature_groups, feature_axis, estimator, scoring, cv, 
               score_summarizer=np.mean, 
               Xvalid=None, yvalid=None, n_jobs=1, pre_dispatch="2*n_jobs", 
               cvsummarizer=None, save_estimator=False, min_n_features=2):
    """
    return_succeed: str, include "score"
        Return value When search suceed.
        (the actual return value: eval(return_succeed))
    return_failed: str 
        Return value When search failed.
        (the actual return value: eval(return_failed))
    """
    if not isinstance(return_succeed, str):
        raise TypeError("return_failed must be str")
    if not isinstance(return_succeed, str):
        raise TypeError("return_failed must be str")
    if "score" not in return_succeed:
        raise Exception("return_succeed must include \"score\"")

    n_splits_ = cv.get_n_splits()
    cvs = cvsummarizer

    def obj(params):
        start_time = datetime.now()
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
                                        X_shape=np.compress(feature_select_ind, X, axis=feature_axis).shape, 
                                        start_time=start_time, end_time=end_time, 
                                        train_score=np.nan, validation_score=np.nan)
                return eval(return_failed)

        # cross validation
        cv_train_scores = []
        cv_test_scores = []
        fit_times = []
        score_times = []

        ret_p = Parallel(
            n_jobs=n_jobs, pre_dispatch=pre_dispatch, 
        )(delayed(fit_and_score)(estimator=clone_estimator(estimator, estimator_params), 
                                 X=np.compress(feature_select_ind, X, axis=feature_axis),  
                                 y=y, 
                                 train_ind=train_ind, test_ind=test_ind, 
                                 scoring=scoring, 
                                 ret_estimator=save_estimator)
        for train_ind, test_ind in cv.split(np.compress(feature_select_ind, X, axis=feature_axis), y, groups))

        # evaluate validation data
        if Xvalid is None:
            train_score, validation_score = np.nan, np.nan
        else:
            train_score, validation_score, _, _, estimator_test = fit_and_score(
                estimator=clone_estimator(estimator, estimator_params), 
                X=np.compress(feature_select_ind, X, axis=feature_axis),  
                y=y, 
                test_data=(np.compress(feature_select_ind, Xvalid, axis=feature_axis), yvalid), 
                scoring=scoring, 
                ret_estimator=True)

        # summarize
        if save_estimator:
            estimators_cv = []
            cnt = 0
            path = os.path.join(cvs.logdir, "estimators"+"_"+cvs.model_id)
            mk_dir(path, error=False)
            name_prefix = cvs.model_id + "_index" + "{0:05d}".format(len(cvs.cv_results_["params"]))
            for i, j, k, l, m in ret_p:
                cv_train_scores.append(i)
                cv_test_scores.append(j)
                fit_times.append(k)
                score_times.append(l)

                dump(m, os.path.join(path, name_prefix+"_split"+"{0:02d}".format(cnt)+".pkl"))
                cnt+=1
            if Xvalid is not None:
                dump(estimator_test, os.path.join(path, name_prefix+"_test"+".pkl"))
        else:
            for i, j, k, l in ret_p:
                cv_train_scores.append(i)
                cv_test_scores.append(j)
                fit_times.append(k)
                score_times.append(l)

        if cvs is not None:
            end_time = datetime.now()
            cvs.store_cv_result(cv_train_scores=cv_train_scores, cv_test_scores=cv_test_scores, params=params, 
                                fit_times=fit_times, score_times=score_times, feature_select=feature_select,
                                X_shape=np.compress(feature_select_ind, X, axis=feature_axis).shape,
                                start_time=start_time, end_time=end_time, 
                                train_score=train_score, validation_score=validation_score)
        score = score_summarizer(cv_test_scores)
        return eval(return_succeed)
    return obj
