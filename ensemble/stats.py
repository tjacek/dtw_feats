import numpy as np
import ensemble.votes,ensemble.single_cls
import utils

class Experiment(object):
    def __init__(self,deep_feats=0):
        if(type(deep_feats)==int):
            self.build_dataset=ensemble.votes.EarlyPreproc(True,None,deep_feats)
        else:
            self.build_dataset=deep_feats
        self.stats={'true_positives':count_agree,'common_errors':indep_rating}

    def __call__(self,deep_paths):
        y_true,all_preds=self.build_dataset.all_predictions(None,deep_paths)
        return { name_i: stat_i(y_true,all_preds)
                    for name_i,stat_i in self.stats.items()}

def show_stats(stats):
    for name_i,value_i in stats.items():
        median_i=np.median(value_i)
        mean_i=np.mean(value_i)
        print("stats:%s median:%f avg:%f" % (name_i,median_i,mean_i)) 

def quality_rating(y_true,all_preds):
    error_matrix=common_errors(y_true,all_preds)
    return np.sum(error_matrix,axis=0)

def indep_rating(y_true,all_preds):
    error_matrix=common_errors(y_true,all_preds)
    n_cls=len(all_preds)
    for i in xrange(n_cls):
        error_matrix[i][i]=0
    return np.sum(error_matrix,axis=0)
                    
def common_errors(y_true,all_preds):
    false_pos= [compare_pred(y_true,pred_i) for pred_i in all_preds]
    def error_helper(j,pred_i,pred_j):
        common_ij=1-compare_pred(pred_i,pred_j)
        return np.dot(common_ij,false_pos[j])
    errors=[[ error_helper(j,pred_i,pred_j)
                for pred_i in all_preds]
                    for j,pred_j in enumerate(all_preds)]
    return np.array(errors) /  float(len(y_true))

def count_agree(single_pred,all_preds):
    size=float(len(single_pred))
    return [ sum(compare_pred(single_pred,pred_i))/size
                for pred_i in all_preds]

def compare_pred(a,b,as_int=True):
    comp=[ a_i!=b_i for a_i,b_i in zip(a,b)]
    if(as_int):
        comp=np.array(comp).astype(int)
    return comp

def cls_compare(y_true,all_preds):
    return [count_agree(pred_i,all_preds) for pred_i in all_preds]