import os, sys, warnings, time, copy
import pandas as pd, numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

from bokeh import layouts
from bokeh.io import output_file, output_notebook, push_notebook, show, save
from bokeh.plotting import figure, ColumnDataSource
from bokeh.resources import INLINE
from bokeh.models import HoverTool, SaveTool, WheelZoomTool, ResetTool, PanTool, BoxZoomTool, LabelSet, CustomJS
from bokeh.models.ranges import DataRange1d, FactorRange
from bokeh.models.widgets import Div

from ._base import mk_dir
from ..model_selection import _setting as st

class CVSummarizer:
    """
    Summarize cross validation results.

    Parameters
    ----------
    cvsize: int.
        Number of folds.

    valid: bool
        Flag whether validation data is input  or not.

    sign: 1 or -1.
        Attribute of sklearn.metrics.make_scorer .
        Flag whether greater is better or not.
    """
    def __init__(self, paraname_list, cvsize, score_summarizer, score_summarizer_name, valid, 
                 sign, model_id, verbose, save_estimator, logdir=None):
        self.score_summarizer = score_summarizer
        self.score_summarizer_name = str(score_summarizer_name)
        self.valid = valid
        self.sign = sign
        self.model_id = str(model_id)

        self.verbose = verbose
        self.save_estimator = save_estimator
        self.logdir = logdir
        if self.logdir is None:
            self.save_path = None
            self.save_graph_path = None
        else:
            path = os.path.join(self.logdir, "cv_results")
            mk_dir(path, error_level=0)
            self.save_path = os.path.join(path, str(self.model_id)+".csv")

            path = os.path.join(self.logdir, "cv_results_graph")
            mk_dir(path, error_level=0)
            self.save_graph_path = os.path.join(path, str(self.model_id)+".html")
            
            if (save_estimator > 0):
                mk_dir(os.path.join(self.logdir, "estimators", self.model_id), 
                    error_level=1, msg="save in this directory.")
        

        self.params_keys = ["param_" + str(i) for i in paraname_list]
        self.train_score_keys = ["split"+str(i)+"_train_score" for i in range(cvsize)]
        self.test_score_keys = ["split"+str(i)+"_test_score" for i in range(cvsize)]
        self.cv_results_ = OrderedDict({"index":[], "params":[]})
        self.next_elapsed_time = np.nan
        self.nbv = None

        self.best_params_ = None
        self.best_score_ = np.nan

    def __call__(self):
        return self.cv_results_

    def _store(self, key, value):
        if key in self.cv_results_:
            self.cv_results_[key].append(value)
        else:
            self.cv_results_[key] = [value]

    def _save(self):
        if self.logdir is not None:    
            if len(pd.DataFrame(self.cv_results_)) == 1:
                if os.path.isfile(self.save_path):
                    warnings.warn("A log file(%s) is already exist. cv result is append to this file" %self.save_path)
            pd.DataFrame(self.cv_results_).iloc[[-1]].to_csv(self.save_path, index=False, encoding="cp932", 
                                                             mode="a", header=(len(pd.DataFrame(self.cv_results_))==1))

    def _init_score(self, cv_train_scores, cv_test_scores, train_score, validation_score):
        if self.sign == 1:
            return cv_train_scores, cv_test_scores, train_score, validation_score
        else:
            cv_train_scores = list(-1*np.array(cv_train_scores))
            cv_test_scores = list(-1*np.array(cv_test_scores))
            train_score = -1*train_score
            validation_score = -1*validation_score
            return cv_train_scores, cv_test_scores, train_score, validation_score

    def _update_best(self):
        if not np.isnan(self.cv_results_[self.score_summarizer_name+"_test_score"]).all():
            if self.sign == 1:
                index = np.nanargmax(self.cv_results_[self.score_summarizer_name+"_test_score"])
            else:
                index = np.nanargmin(self.cv_results_[self.score_summarizer_name+"_test_score"])

            self.best_params_ = self.cv_results_["params"][index]
            self.best_score_ = self.cv_results_[self.score_summarizer_name+"_test_score"][index]
            
    def store_cv_result(self, cv_train_scores, cv_test_scores, params, fit_times, score_times, 
                        feature_select, X_shape, start_time,
                        end_time, train_score, validation_score):
        cv_train_scores, cv_test_scores, train_score, validation_score = self._init_score(cv_train_scores, cv_test_scores, train_score, validation_score)

        # Summary
        self._store("index", len(self.cv_results_["index"]))
        self._store("params", params)
        self._store("start_time", start_time)
        self._store("end_time", end_time)

        self._store(self.score_summarizer_name+"_train_score", self.score_summarizer(cv_train_scores))
        self._store("std_train_score", np.std(cv_train_scores))
        self._store(self.score_summarizer_name+"_test_score", self.score_summarizer(cv_test_scores))
        self._store("std_test_score", np.std(cv_test_scores))
        self._store("train_score(whole)", train_score)
        self._store("validation_score", validation_score)

        # Score details
        for i , key in enumerate(self.train_score_keys):
            self._store(key, cv_train_scores[i])
        for i , key in enumerate (self.test_score_keys):
            self._store(key, cv_test_scores[i])

        # Parameter details
        self._store("X_shape", X_shape)
        self._store("feature_select", feature_select)
        for key in self.params_keys:
            self._store(key, params[key.split("param_")[1]])

        # Time details
        self._store("mean_fit_time", np.mean(fit_times))
        self._store("mean_score_time", np.mean(score_times))
        self._store("std_fit_time", np.std(fit_times))
        self._store("std_score_time", np.std(score_times))
        self._store("elapsed_time_sec(estimated)", self.next_elapsed_time)
        if isinstance(start_time, datetime) & isinstance(end_time, datetime):
            self._store("elapsed_time_sec", (end_time-start_time).seconds)
        else:
            self._store("elapsed_time_sec", np.nan)

        self._store("model_id", self.model_id)
        self._save()
        self._update_best()

    def _estimate_time_sec(self, params):
        df = pd.DataFrame(self.cv_results_["params"]+[params])

        strcols = df.columns[df.dtypes==object].tolist()
        df[strcols] = df[strcols].fillna("none")
        le = LabelEncoder()
        for strcol in strcols:
            df.loc[:, strcol] = le.fit_transform(df.loc[:, strcol].tolist())

        Xtrain = df.iloc[:df.index.max()].dropna(axis=0, inplace=False)
        Xtest = df.iloc[[df.index.max()]].dropna(axis=0, inplace=False)

        if (len(Xtrain) ==0) or (len(Xtest)==0):
            self.next_elapsed_time = np.nan
        else:
            estimator = RandomForestRegressor()
            estimator.fit(Xtrain, self.cv_results_["elapsed_time_sec"])
            self.next_elapsed_time = int(estimator.predict(Xtest))

    def display_status(self, params, start_time=None):
        if self.verbose > 0:
            self._estimate_time_sec(params)
            if start_time is None:
                start_time = datetime.now()

            n_search = len(self.cv_results_["params"])
            if np.isnan(self.next_elapsed_time):
                estimated_end_time = np.nan
            else:
                estimated_end_time = start_time + timedelta(seconds=self.next_elapsed_time)
                estimated_end_time = estimated_end_time.strftime("%m/%d %H:%M")

            if self.verbose == 1:
                start_time = start_time.strftime("%m/%d %H:%M")
                sys.stdout.write("\rNum_of_search:%s  Start:%s  End(estimated):%s  Best_score:%s" 
                                %(n_search, start_time, estimated_end_time, np.round(self.best_score_, 2)))
            elif self.verbose == 2:
                if self.nbv is None:
                    if n_search > 0:
                        self.nbv = NoteBookVisualizer(cv_results_cols=self.cv_results_.keys(), sign=self.sign, valid=self.valid, 
                                                      savepath=self.save_graph_path)
                else:
                    self.nbv.fit(cv_results=self.cv_results_, estimeted_end_time=estimated_end_time)
            
            

class NoteBookVisualizer():
    """
    Visualize cross validation results.
    """
    time_col = "end_time"
    score_cols = dict(train="mean_train_score", test="mean_test_score", valid="validation_score")
    score_std_cols = dict(train="std_train_score", test="std_test_score")
    colors = dict(train="#1f77b4", test="#ff7f0e", valid="#2ca02c")
    display_width = 950
    n_col_param = 5
    stream_rollover = 256
    headline = """
               <div style="font-family:segoe ui, sans-serif; font-style:italic; font-size:x-large; 
               border-bottom:solid 2.5px #7f7f7f; color:#1987E5; padding-bottom: 3px;">TEXT</div>
               """

    def _update_cv_score_std_src(self, cv_score_std):
        patches = dict()
        for key in cv_score_std.keys():
            patches[key] = [(slice(NoteBookVisualizer.stream_rollover*2), cv_score_std[key])]
        self.cv_score_std_src.patch(patches)
        push_notebook(handle=self.bokeh_handle)
            
    def _update_param_srcs(self, param_dists):
        for key in param_dists.keys():
            new_data = dict()
            patches = dict()
            for inner_key in list(self.param_srcs[key].data.keys()):
                old_len = len(self.param_srcs[key].data[inner_key])
                new_len = len(param_dists[key][inner_key])
                patches[inner_key] = [(slice(old_len), param_dists[key][inner_key][:old_len])]
                if old_len == new_len:
                    pass
                elif old_len < new_len:
                    new_data[inner_key] = param_dists[key][inner_key][old_len:]
                else:
                    raise Exception("Inner Error: param_dists[", key, "] 's length must not decrease.")
            self.param_srcs[key].patch(patches)

            if len(new_data) > 0:
                self.param_srcs[key].stream(new_data)
            push_notebook(handle=self.bokeh_handle)
            
    def _mk_partial_dict(self, tgt_dict, tgt_keys):
        return dict(zip(tgt_keys, [tgt_dict[key] for key in tgt_keys]))
    
    def _mk_score_source(self, cv_results, xcol, score_cols, hover_cols=None):
        if isinstance(cv_results, pd.core.frame.DataFrame):
            cv_results = cv_results.to_dict(orient="list")
        src_dict = self._mk_partial_dict(tgt_dict=cv_results, tgt_keys=[xcol]+score_cols)
        if hover_cols is None:
            return ColumnDataSource(src_dict)
        else:
            tooltips = []
            for col in hover_cols:
                tooltips.append((str(col) ,"@"+str(col)))
                src_dict[col] = cv_results[col]
            return ColumnDataSource(src_dict), HoverTool(tooltips=tooltips)
        

    def _add_line(self, p, xcol, ycol, score_source, score_std_source=None, color="black", legend=None):
        if score_std_source is not None:
            p.patch(x=xcol, y=ycol, fill_alpha=0.5, line_alpha=0, source=score_std_source, 
                    fill_color=color, line_color=color, legend=legend)  
        p.line(x=xcol, y=ycol, source=score_source,  
               line_color=color, legend=legend, line_width=1)
        p.circle(x=xcol, y=ycol, source=score_source, 
                 line_color=color, legend=legend, fill_color="white", size=4)
        return p

    def _arrange_fig(self, p):
        p.toolbar.logo = None
        p.xaxis.axis_label_text_font = "segoe ui"
        p.yaxis.axis_label_text_font = "segoe ui"
        p.yaxis.axis_label_text_font_style = "normal"
        p.axis.major_label_text_font = "segoe ui"
        p.legend.click_policy="hide"
        return p
    
    def _init_cv_results(self, cv_results):
        cv_results = pd.DataFrame(cv_results)
        cv_results = cv_results[~cv_results[[NoteBookVisualizer.time_col, NoteBookVisualizer.score_cols["test"], NoteBookVisualizer.score_std_cols["test"]]].isnull().any(axis=1)]

        cv_results.reset_index(drop=True, inplace=True)
        if len(cv_results) == 0:
            return None, None, None

        cv_results.rename(columns=dict([(col, col.split("param_")[-1]) for col in cv_results.columns if "param_" in col]), inplace=True)

        if self.sign == 1:
            cv_results[["best_train", "best_test", "best_valid"]] = cv_results[[NoteBookVisualizer.score_cols["train"], NoteBookVisualizer.score_cols["test"], NoteBookVisualizer.score_cols["valid"]]].cummax()
        else:
            cv_results[["best_train", "best_test", "best_valid"]] = cv_results[[NoteBookVisualizer.score_cols["train"], NoteBookVisualizer.score_cols["test"], NoteBookVisualizer.score_cols["valid"]]].cummin()
               
        cv_score_std = {NoteBookVisualizer.time_col:cv_results[NoteBookVisualizer.time_col].tolist()+cv_results[NoteBookVisualizer.time_col].tolist()[::-1],}
        for data_type in ["train", "test"]:
            cv_score_std[self.score_cols[data_type]] = (cv_results[self.score_cols[data_type]]+cv_results[self.score_std_cols[data_type]]).tolist()
            cv_score_std[self.score_cols[data_type]] += (cv_results[self.score_cols[data_type]]-cv_results[self.score_std_cols[data_type]]).iloc[::-1].tolist()
        
        # to support stream_rollover
        css_length = int(len(cv_score_std[NoteBookVisualizer.time_col])/2)
        if css_length > NoteBookVisualizer.stream_rollover:
            for key in cv_score_std.keys():
                cv_score_std[key] = cv_score_std[key][css_length - NoteBookVisualizer.stream_rollover :css_length + NoteBookVisualizer.stream_rollover]
        else:
            for key in cv_score_std.keys():
                cv_score_std[key].extend(["nan"]*((NoteBookVisualizer.stream_rollover - css_length)*2))

        param_dists = dict()
        if len(self.param_feature_cols) > 1:
            param_dists[st.FEATURE_SELECT_PARAMNAME_PREFIX] = dict(label=[i.split(st.FEATURE_SELECT_PARAMNAME_PREFIX)[-1] for i in self.param_feature_cols], 
                                                                    x=[int(i.split(st.FEATURE_SELECT_PARAMNAME_PREFIX)[-1])-1 for i in self.param_feature_cols], 
                                                                    top=cv_results[self.param_feature_cols].sum(0).values.tolist())
        for param_col in self.param_cols:
            if cv_results[param_col].dtypes == "object":
                vc = cv_results[param_col].value_counts(dropna=False).sort_index()
                obj_param_dist = dict(label=vc.index.fillna("none").tolist(), top=vc.values.tolist())
                try:
                    chk_dict = dict(zip(self.param_srcs["label"], self.param_srcs["x"]))
                    for label in list(obj_param_dist["label"]):
                        if not(label in self.param_srcs["label"]):
                            chk_dict[label] = len(chk_dict[label]) - 1

                    obj_param_dist["label"] = list(chk_dict.keys())
                    obj_param_dist["x"] = list(chk_dict.values())                        
                except (AttributeError, KeyError):
                    obj_param_dist["x"] = [i for i in range(len(obj_param_dist["label"]))]
                param_dists[param_col] = copy.deepcopy(obj_param_dist)
            else:
                hist, edges = np.histogram(cv_results[param_col], density=False, bins=10)
                param_dists[param_col] = dict(left=list(edges[:-1]), right=list(edges[1:]), top=list(hist))
                    
        cv_results[self.param_cols] = cv_results[self.param_cols].fillna("none")
        
        return cv_results, cv_score_std, param_dists
    
    def __init__(self, cv_results_cols, sign, valid, savepath):
        if valid:
            self.data_types = ["train", "test", "valid"]
        else:
            self.data_types = ["train", "test"]

        self.param_feature_cols = [i.split("param_")[-1] for i in cv_results_cols if("param_"+st.FEATURE_SELECT_PARAMNAME_PREFIX in i)&(i!="param_"+st.FEATURE_SELECT_PARAMNAME_PREFIX+str(st.ALWAYS_USED_FEATURE_GROUP_ID))]
        self.all_param_cols = [i.split("param_")[-1] for i in cv_results_cols if("param_" in i)&(i!="param_"+st.FEATURE_SELECT_PARAMNAME_PREFIX+str(st.ALWAYS_USED_FEATURE_GROUP_ID))]
        self.param_cols = list(set(self.all_param_cols)-set(self.param_feature_cols))
        self.param_cols.sort()
        
        self.sign = sign
        self.cv_results_cols = cv_results_cols
        self.valid = valid
        self.savepath = savepath

        self.bokeh_handle = None
        
    def fit(self, cv_results, estimeted_end_time):
        cv_results, cv_score_std, param_dists = self._init_cv_results(cv_results)

        if self.bokeh_handle is None:
            if cv_results is None:
                return

            # mk bokeh source
            self.cv_src, cv_hover = self._mk_score_source(cv_results, xcol=NoteBookVisualizer.time_col, score_cols=[NoteBookVisualizer.score_cols[i] for i in self.data_types], 
                                                          hover_cols=self.all_param_cols)
            self.end_time_src = ColumnDataSource(data=dict(text=["This search end time(estimated): "+estimeted_end_time]))
            self.cv_score_std_src = ColumnDataSource(data=cv_score_std)
            self.best_src = self._mk_score_source(cv_results, xcol=NoteBookVisualizer.time_col, score_cols=["best_"+i for i in self.data_types])
            
            self.param_srcs = dict()
            for key in param_dists.keys():
                self.param_srcs[key] = ColumnDataSource(data= param_dists[key])
       

            # CV Score transition
            cv_p = figure(title="CV Score transition", x_axis_label="time", y_axis_label="score", 
                          x_axis_type="datetime", plot_width=int(NoteBookVisualizer.display_width/2), plot_height=275, 
                          toolbar_location="above", 
                          tools=[SaveTool(), ResetTool(), PanTool(), WheelZoomTool()])

            for data_type in self.data_types:
                if data_type=="valid":
                    cv_p = self._add_line(cv_p, xcol=NoteBookVisualizer.time_col,  ycol=NoteBookVisualizer.score_cols[data_type], 
                                          score_source=self.cv_src, 
                                          color=NoteBookVisualizer.colors[data_type], legend=data_type)
                else:
                    cv_p = self._add_line(cv_p, xcol=NoteBookVisualizer.time_col, ycol=NoteBookVisualizer.score_cols[data_type], 
                                          score_source=self.cv_src, score_std_source=self.cv_score_std_src, 
                                          color=NoteBookVisualizer.colors[data_type], legend=data_type)

            display_etime = LabelSet(x=0, y=0, x_offset=80, y_offset=20, 
                                     x_units="screen", y_units="screen", render_mode="canvas",
                                     text="text", source=self.end_time_src, 
                                     text_font="segoe ui", text_font_style ="italic", 
                                     background_fill_color="white", background_fill_alpha=0.5)
            cv_p.add_layout(display_etime)

            cv_p.add_tools(cv_hover)
            cv_p.legend.location = "top_left"
            cv_p.xaxis.minor_tick_line_color = None
            cv_p.yaxis.minor_tick_line_color = None
            cv_p = self._arrange_fig(cv_p)
            
            
            # Best Score transition
            best_p = figure(title="Best Score transition", x_axis_label="time", y_axis_label="score", 
                            x_range=cv_p.x_range, y_range=cv_p.y_range, 
                            x_axis_type="datetime", plot_width=int(NoteBookVisualizer.display_width/2), plot_height=275, 
                            toolbar_location="above", tools=[PanTool(), WheelZoomTool(), SaveTool(), ResetTool()])
            for data_type in self.data_types:
                best_p = self._add_line(best_p, xcol=NoteBookVisualizer.time_col, ycol="best_"+data_type, 
                                       score_source=self.best_src, color=NoteBookVisualizer.colors[data_type], legend=data_type)
            best_p.legend.location = "top_left"
            best_p.xaxis.minor_tick_line_color = None
            best_p.yaxis.minor_tick_line_color = None
            best_p = self._arrange_fig(best_p)

            
            # Param distributions
            param_vbar_ps = dict()
            param_hist_ps = dict()

            tmp = list(self.param_cols)
            if st.FEATURE_SELECT_PARAMNAME_PREFIX in self.param_srcs.keys():
                tmp = [st.FEATURE_SELECT_PARAMNAME_PREFIX] + tmp
            for param_col in tmp:
                if "label" in list(param_dists[param_col].keys()):
                    # Bar graph
                    param_vbar_ps[param_col] = figure(title=param_col, y_axis_label="frequency", 
                                                 plot_width=int(NoteBookVisualizer.display_width/NoteBookVisualizer.n_col_param), 
                                                 plot_height=int(NoteBookVisualizer.display_width/NoteBookVisualizer.n_col_param), 
                                                 #x_range=FactorRange(factors=self.param_srcs[param_col].data["x"]), 
                                                 y_range=DataRange1d(min_interval=1.0, start=0, default_span=1.0), 
                                                 toolbar_location="above", 
                                                 tools=[SaveTool(), HoverTool(tooltips=[("label","@label"), ("top","@top")])])
                    param_vbar_ps[param_col].vbar(x="x", top="top", 
                                             source=self.param_srcs[param_col], 
                                             width=0.5, bottom=0, color="#9467bd", fill_alpha=0.5)

                    labels = LabelSet(x="x", y=0, level="glyph", text="label", text_align="center", 
                                      text_font="segoe ui", text_font_style="normal", text_font_size="8pt", 
                                      x_offset=0, y_offset=0, source=self.param_srcs[param_col], render_mode="canvas")
                    param_vbar_ps[param_col].add_layout(labels)

                    param_vbar_ps[param_col].xaxis.major_label_text_font_size = "0pt"
                    param_vbar_ps[param_col].xaxis.major_tick_line_color = None
                    param_vbar_ps[param_col].xaxis.minor_tick_line_color = None
                    param_vbar_ps[param_col].yaxis.minor_tick_line_color = None
                    param_vbar_ps[param_col] = self._arrange_fig(param_vbar_ps[param_col])
                else: 
                    # Histgram
                    param_hist_ps[param_col] = figure(title=param_col, y_axis_label="frequency", 
                                                 plot_width=int(NoteBookVisualizer.display_width/NoteBookVisualizer.n_col_param), 
                                                 plot_height=int(NoteBookVisualizer.display_width/NoteBookVisualizer.n_col_param), 
                                                 y_range=DataRange1d(min_interval=1.0, start=0), 
                                                 toolbar_location="above", 
                                                 tools=[SaveTool(), HoverTool(tooltips=[("left","@left"), ("right","@right"), ("top","@top")])])
                    param_hist_ps[param_col].quad(top="top", bottom=0, left="left", right="right", 
                                             source=self.param_srcs[param_col], 
                                             color="#17becf", fill_alpha=0.5)
                    param_hist_ps[param_col].xaxis.minor_tick_line_color = None 
                    param_hist_ps[param_col].yaxis.minor_tick_line_color = None 
                    param_hist_ps[param_col] = self._arrange_fig(param_hist_ps[param_col])
                    

            scores_headline = Div(text=NoteBookVisualizer.headline.replace("TEXT", " Score History"), width=int(NoteBookVisualizer.display_width*0.9))
            params_headline = Div(text=NoteBookVisualizer.headline.replace("TEXT", " Parameter History"), width=int(NoteBookVisualizer.display_width*0.9))
            self.p = layouts.layout([[scores_headline]]+[[cv_p, best_p]]+[[params_headline]]+\
                               [list(param_vbar_ps.values())[i:i+NoteBookVisualizer.n_col_param] for i in range(0, len(param_vbar_ps), NoteBookVisualizer.n_col_param)]+\
                               [list(param_hist_ps.values())[i:i+NoteBookVisualizer.n_col_param] for i in range(0, len(param_hist_ps), NoteBookVisualizer.n_col_param)])
            self.bokeh_handle = show(self.p, notebook_handle=True)
        else:
            # update bokeh src
            self.end_time_src.patch({"text":[(0, "This search end time(estimated): "+estimeted_end_time)]})
            if len(cv_results) != len(self.cv_src.data[NoteBookVisualizer.time_col]):
                self.cv_src.stream(cv_results[list(self.cv_src.data.keys())].iloc[-1:].to_dict(orient="list"), 
                                   rollover=NoteBookVisualizer.stream_rollover)
                self.best_src.stream(cv_results[list(self.best_src.data.keys())].iloc[-1:].to_dict(orient="list"), 
                                     rollover=NoteBookVisualizer.stream_rollover)
                push_notebook(handle=self.bokeh_handle)

                self._update_cv_score_std_src(cv_score_std)
                self._update_param_srcs(param_dists)

            if self.savepath is not None:
                save(self.p, filename=self.savepath, resources=INLINE)

