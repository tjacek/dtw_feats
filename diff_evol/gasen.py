import sys
sys.path.append("..")
import numpy as np
from scipy.optimize import differential_evolution
import exp,ens,k_fold,learn

class Corl(object):
    def __init__(self,all_votes):
        self.all_votes=ens.Votes(all_votes)	
        self.d=[ result_i.true_one_hot() 
                  for result_i in all_votes]
    
    def __call__(self,weights):
        norm=weights/np.sum(weights)
        C=self.corl()
        n_clf=len(self.all_votes)
        loss=0
        for i in range(n_clf):
            for j in range(n_clf):
                loss+=weights[i]*weights[j] * C[i,j] 	
        return loss

    def corl(self):
        results=self.all_votes.results
        n_clf=len(self.all_votes)
        C=np.zeros((n_clf,n_clf))
        for i in range(n_clf):
            for j in range(n_clf):
                f_i=results[i].y_pred
                f_j=results[j].y_pred
                c_ij= (f_i-self.d)*(f_j-self.d)
                C[i,j]=np.mean(c_ij)
        return C

class OptimWeights(object):
    def __init__(self, loss=None):
        if(loss is None):
            loss=Corl
        self.loss = loss

    def __call__(self,common,deep,clf="LR"):
        datasets=ens.read_dataset(common,deep)
        weights=self.find_weights(datasets)
        results=learn.train_ens(datasets,clf)
        votes=ens.Votes(results)
        result=votes.weighted(weights)
        return result

    def find_weights(self,datasets,clf="LR"):
        results=self.validation_votes(datasets,clf)
        loss_fun=self.loss(results)
        bound_w = [(0.01, 1.0)  for _ in datasets]
        result = differential_evolution(loss_fun, bound_w, 
    			maxiter=10, tol=1e-7)
        weights=result['x']
        return weights

    def validation_votes(self,datasets,clf="LR"):
        train=[data_i.split()[0] for data_i in datasets]
        names=list(train[0].keys())
        selector_gen=k_fold.StratGen(1)
        selector=list(selector_gen(names))[0]
        return learn.train_ens(train,clf=clf,selector=selector)

dataset="3DHOI"
dir_path="../../ICSS"#%s" % dataset
paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
print(paths)
optim=OptimWeights(Corl)
result=optim(paths["common"],paths["binary"])
result.report()