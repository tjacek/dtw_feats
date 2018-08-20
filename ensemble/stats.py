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
        self.stats={'common_errors':IndepMetric(True),'common_preds':IndepMetric(False),
                    'diversity_rating':diversity_rating,'individual_accuracy':true_pos,
                    'ensemble_accuracy':ensemble_accuracy,'votes':vote_histogram}

    def __call__(self,deep_paths,basic_paths=None):
        y_true,all_preds=self.build_dataset.all_predictions(basic_paths,deep_paths)
        return { name_i: stat_i(y_true,all_preds)
                    for name_i,stat_i in self.stats.items()}

class IndepMetric(object):
    def __init__(self,error):
        self.error=error

    def __call__(self,y_true,all_preds):    
        def indep_helper(pred_i):   
            mask_vector=compare_pred(y_true,pred_i,erorr=self.error)
            comp_vectors=compare_all(pred_i,all_preds,erorr=False)
            return [np.dot(mask_vector,comp_vect_i)  
                    for comp_vect_i in comp_vectors]
        indep_matrix=np.array([ indep_helper(pred_i) 
                                for pred_i in all_preds])
#        np.fill_diagonal(indep_matrix, 0)
#        indep_matrix/=float(len(y_true))
        return indep_matrix#np.median(indep_matrix,axis=0)
       
def show_stats(stats):
    for name_i,value_i in stats.items():
        if(isinstance(value_i,(int,float,long))):
            print("stats:%s %f" % (name_i,value_i))
            break
        if(value_i.ndim==1):
            median_i=np.median(value_i)
            mean_i=np.mean(value_i)
            max_i=np.amax(value_i)
            min_i=np.amin(value_i)
            std_i=np.std(value_i)
            all_stats=(name_i,median_i,mean_i,std_i,max_i,min_i)
            print("stats:%s median:%f avg:%f std:%f max:%f min:%f" % all_stats) 

def best_clf(stats,k=5,criterion='individual_accuracy'):
    clf_metric=stats[criterion]
    return np.argsort(clf_metric)[:k]

def ensemble_accuracy(y_true,all_preds):
    y_pred=ensemble.votes.vote(all_preds)
    return accuracy_score(y_true,y_pred)

def diversity_rating(y_true,all_preds):
    diversity=np.array([true_pos(pred_j,all_preds)
                            for pred_j in all_preds])
    diversity=np.mean(diversity,axis=0)
    return diversity

def vote_histogram(y_true,all_preds):
    true_pos=compare_all(y_true,all_preds,erorr=False)
    votes=np.sum(true_pos,axis=0)
    n_cats=true_pos.shape[0]+1
    hist=np.zeros((n_cats,))
    for vote_i in votes:
        hist[vote_i]+=1.0
    hist/=np.sum(hist)    
    return hist

def true_pos(y_true,all_preds):
    return [accuracy_score(y_true,pred_i)
                for pred_i in all_preds]

def compare_all(pred,list_of_preds,erorr=False):
    return np.array([compare_pred(pred,pred_i,erorr=erorr) 
                        for pred_i in list_of_preds])
    
def compare_pred(a,b,erorr=False):
    comp=np.array([ int(a_i==b_i) for a_i,b_i in zip(a,b)])
    if(erorr):
        comp=1-comp
    return comp