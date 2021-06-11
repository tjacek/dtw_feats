import sys
sys.path.append("..")    
import numpy as np
from scipy.optimize import differential_evolution
import ens,auc,exp,learn,diff

class ScoreLoss(object):
    def __init__(self,results):
        self.results=results
        self.iter=0

    def __call__(self,score_weights):
        self.iter+=1
        print(self.iter)
        score_weights=score_weights/np.sum(score_weights)
        votes=score_dataset(self.results,score_weights)
        final_result=votes.voting(False)
        return diff.mse_fun(final_result)

def score_dataset(results,score_weights):
    pref=[as_pref(result_i,score_weights) 
                for result_i in results]
    y_true,names=results[0].y_true,results[0].names
    results=[learn.Result(y_true,pref_i,names) 
                    for pref_i in pref]
    return ens.Votes(results)

def score_opt(paths,clf="LR"):
    datasets=ens.read_dataset(paths['common'],paths['binary'])
    val=auc.CrossVal(0.5)
    new_datasets,results=val(datasets,clf)
    weights=find_weights(results)
    results=learn.train_ens(datasets)
    votes=score_dataset(results,weights)
    final_result=votes.voting(False)
    final_result.report()

def find_weights(results):
#    score_weights=borda_weights(len(results))
    loss_fun=ScoreLoss(results)       
    bound_w = [(0.01, 1)#np.amax(score_weights))  
                for _ in results]
    popsize=15
#    init_matrix=np.array([score_weights 
#                    for i in range(popsize)])
    result = differential_evolution(loss_fun, bound_w, 
                popsize=popsize,maxiter=10, tol=1e-7)#,
#                init=init_matrix)
    weights=result['x']
    return weights

def borda_weights(n_cats):
    weights=[float(i) for i in range(n_cats)]
    weights.reverse()
    return np.array( weights)

def as_pref(result_i,score_weights):
    X=result_i.as_numpy()
    new_X=[]
    for x_i in X:
        ind_i=np.argsort(x_i)#np.flip(np.argsort(x_i))
        new_x_i=np.zeros(ind_i.shape)
        for i,w_i in enumerate(ind_i):
        	new_x_i[w_i]=score_weights[i]
        new_X.append(new_x_i)
    return np.array(new_X)

if __name__ == "__main__":
    dataset="3DHOI"
    dir_path="../../ICSS"#%s" % dataset
    paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
    paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
    score_opt(paths,clf="LR")