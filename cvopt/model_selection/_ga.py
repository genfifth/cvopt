import numpy as np

from ..search_setting._base import get_params, to_func

def eval_fitness(scores, sign):
    """
    Fitness is in proportion to difference from worst score.
    """
    scores = sign*scores
    scores = np.nanmax(scores) - scores
    scores /= np.nansum(scores)
    scores[np.isnan(scores)] = 0
    return scores

    
def crossover(fitness, param_crossover_proba, cv_results, start_index):
    parents_index = np.random.choice(len(fitness), size=2, replace=False, p=fitness) + start_index
    parents_params_base = cv_results["params"][parents_index[0]]
    parents_params_tgt = cv_results["params"][parents_index[1]]
    
    child_params = {}
    for key in parents_params_base.keys():
        if np.random.rand() < param_crossover_proba:
            child_params[key] = parents_params_tgt[key]
        else:
            child_params[key] = parents_params_base[key]
    return child_params


def mutate(params, param_mutation_proba, param_distributions):
    ret = {}
    for key in params.keys():
        if np.random.rand() < param_mutation_proba:
            ret.update(get_params(param_distributions, tgt_key=key))
        else:
            ret[key] = params[key]
    return ret


def gamin(obj, param_distributions, max_iter, iter_pergeneration, param_crossover_proba, param_mutation_proba, random_sampling_proba, cvsummarizer):
    it = 0
    param_crossover_proba = to_func(param_crossover_proba)
    param_mutation_proba = to_func(param_mutation_proba)
    random_sampling_proba = to_func(random_sampling_proba)

    while True:
        # Set GA params.
        population = []
        if it == 0:
            generaion = 0
            rsp = 1
            start_index = None
            fitness = None
            
        else:
            generaion = int((it+1) / iter_pergeneration)
            rsp = random_sampling_proba(generaion)
            start_index = it - iter_pergeneration
            fitness = eval_fitness(scores=np.array(cvsummarizer.cv_results_[cvsummarizer.score_summarizer_name+"_test_score"])[start_index:], 
                                   sign=cvsummarizer.sign)
            

            if (fitness > 0).sum() < 2:
                # If there are not enough parents in this generaion, use old generation scores.
                fitness = eval_fitness(scores=np.array(cvsummarizer.cv_results_[cvsummarizer.score_summarizer_name+"_test_score"]), 
                                       sign=cvsummarizer.sign)
                start_index = 0

                if (fitness > 0).sum() < 2:
                    # If there are not enough parents in all generaion, all next generation is random sample.
                    rsp = 1
        
        # Create population.
        for i in range(iter_pergeneration):
            if np.random.rand() < rsp:
                population.append(get_params(param_distributions, tgt_key=None))
            else:
                child = crossover(fitness=fitness, param_crossover_proba=param_crossover_proba(generaion), 
                                  cv_results=cvsummarizer.cv_results_, start_index=start_index)
                child = mutate(params=child, param_mutation_proba=param_mutation_proba(generaion), 
                               param_distributions=param_distributions)
                population.append(child)
    
        # Do search.
        for params in population:
            obj(params)
            it += 1
            if it >= max_iter:
                return
