import numpy as np
from sklearn.metrics import accuracy_score
import ensemble.votes,ensemble.single_cls
import utils

class Experiment(object):
    def __init__(self,deep_feats=0):
        if(type(deep_feats)==int):
            self.build_dataset=ensemble.votes.EarlyPreproc(True,None,deep_feats)
        else:
            self.build_dataset=deep_feats
        self.stats={'common_errors':IndepMetric(False),'common_prediction':IndepMetric(True),
                    'quality_rating':quality_rating,'individual_accuracy':true_pos,
                    'ensemble_accuracy':ensemble_accuracy}

    def __call__(self,deep_paths,basic_paths=None):
        y_true,all_preds=self.build_dataset.all_predictions(basic_paths,deep_paths)
        return { name_i: stat_i(y_true,all_preds)
                    for name_i,stat_i in self.stats.items()}

class IndepMetric(object):
    def __init__(self,flip):
        self.flip=flip

    def __call__(self,y_true,all_preds):    
        error_matrix=common_preds(y_true,all_preds,flip=self.flip)
        n_cls=len(all_preds)
        for i in xrange(n_cls):
            error_matrix[i][i]=0
        n_cls=float(error_matrix.shape[0])
        return np.sum(error_matrix,axis=0)/n_cls
       
def show_stats(stats):
    for name_i,value_i in stats.items():
        if(isinstance(value_i,(int,float,long))):
            print("stats:%s %f" % (name_i,value_i))
        else:
            median_i=np.median(value_i)
            mean_i=np.mean(value_i)
            max_i=np.amax(value_i)
            min_i=np.amin(value_i)
            all_stats=(name_i,median_i,mean_i,max_i,min_i)
            print("stats:%s median:%f avg:%f max:%f min:%f" % all_stats) 

def ensemble_accuracy(y_true,all_preds):
    y_pred=ensemble.votes.vote(all_preds)
    return accuracy_score(y_true,y_pred)

def quality_rating(y_true,all_preds):
    error_matrix=common_preds(y_true,all_preds)
    n_cls=float(error_matrix.shape[0])
    return np.sum(error_matrix,axis=0)/n_cls

def common_preds(y_true,all_preds,flip=False):
    true_pos= [compare_pred(y_true,pred_i,flip=flip) 
                for pred_i in all_preds]
    def pred_helper(j,pred_i,pred_j):
        common_ij=compare_pred(pred_i,pred_j,flip=True)
        return np.dot(common_ij,true_pos[j])
    preds= [[ pred_helper(j,pred_i,pred_j)
                for pred_i in all_preds]
                    for j,pred_j in enumerate(all_preds)]
    return np.array(preds) /  float(len(y_true))

def true_pos(y_true,all_preds):
    return [accuracy_score(y_true,pred_i)
                for pred_i in all_preds]

def count_agree(single_pred,all_preds):
    size=float(len(single_pred))
    return [ sum(compare_pred(single_pred,pred_i))/size
                for pred_i in all_preds]

def compare_pred(a,b,flip=False):
    comp=[ a_i!=b_i for a_i,b_i in zip(a,b)]
    comp=np.array(comp).astype(int)
    if(flip):
        comp=1-comp
    return comp

def cls_compare(y_true,all_preds):
    return [count_agree(pred_i,all_preds) for pred_i in all_preds]