import numpy as np
import ensemble.votes,ensemble.single_cls
import utils

class Experiment(object):
    def __init__(self,deep_feats=None):
        self.build_dataset=ensemble.votes.EarlyPreproc(True,None,deep_feats)
        self.stats={'true_positives':count_agree,'common_errors':common_errors}

    def __call__(self,deep_path):
        datasets=[self.build_dataset.get_dataset(path_i)
                    for path_i in utils.bottom_files(deep_path)]
        result=[ ensemble.single_cls.train_model(dataset_i)
                    for dataset_i in datasets]
        y_true=result[0][0]
        y_pred=[result_i[1] for result_i in result]
        return { name_i: stat_i(y_true,y_pred)
                    for name_i,stat_i in self.stats.items()}

def cls_compare(y_true,all_preds):
    return [count_agree(pred_i,all_preds) for pred_i in all_preds]
 
def count_agree(single_pred,all_preds):
    size=float(len(single_pred))
    return [ sum(compare_pred(single_pred,pred_i))/size
                for pred_i in all_preds]
                    
def compare_pred(a,b,as_int=True):
    comp=[ a_i!=b_i for a_i,b_i in zip(a,b)]
    if(as_int):
        comp=np.array(comp).astype(int)
    return comp

def common_errors(y_true,all_preds):
    false_pos= [compare_pred(y_true,pred_i) for pred_i in all_preds]
    def error_helper(j,pred_i,pred_j):
        common_ij=1-compare_pred(pred_i,pred_j)
        return np.dot(common_ij,false_pos[j])
    errors=[[ error_helper(j,pred_i,pred_j)
                for pred_i in all_preds]
                    for j,pred_j in enumerate(all_preds)]
    return np.array(errors) /  float(len(y_true))