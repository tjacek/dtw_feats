import sys
sys.path.append("..")
import numpy as np
from scipy.optimize import differential_evolution
import ens,exp,files,learn,k_fold

class LossFunc(object):
    def __init__(self,all_votes):
        self.all_votes=all_votes
        self.d=[ result_i.true_one_hot() 
                  for result_i in all_votes]
        self.n_iters=0

    def __call__(self,weights):
        norm=weights/np.sum(weights)
        C=self.corl()
        n_clf=len(self.all_votes)
        loss=0
        for i in range(n_clf):
            for j in range(n_clf):
                loss+=weights[i]*weights[j] * C[i,j] 
        self.n_iters+=1
        print(self.n_iters)
        print(loss)
        return loss

    def corl(self):
        n_clf=len(self.all_votes)
        C=np.zeros((n_clf,n_clf))
        for i in range(n_clf):
            for j in range(n_clf):
                f_i=self.all_votes[i].y_pred
                f_j=self.all_votes[j].y_pred
                c_ij= (f_i-self.d)*(f_j-self.d)
                C[i,j]=np.mean(c_ij)
        return C

def diff_voting(common,deep,clf="LR"):
    datasets=ens.read_dataset(common,deep)
    weights=find_weights(datasets)#,loss=loss)
    results=learn.train_ens(datasets,clf)
    votes=ens.Votes(results)
    result=votes.weighted(weights)
    return result

def find_weights(datasets,clf="LR"):
    results=validation_votes(datasets,clf)
    loss_fun=LossFunc(results)
    bound_w = [(0.01, 1.0)  for _ in datasets]
    result = differential_evolution(loss_fun, bound_w, 
    			maxiter=10, tol=1e-7)
    weights=result['x']
    return weights

def validation_votes(datasets,clf="LR"):
    train=[data_i.split()[0] for data_i in datasets]
    names=list(train[0].keys())
    selector_gen=k_fold.StratGen(1)
    selector=list(selector_gen(names))[0]
    return learn.train_ens(train,clf=clf,selector=selector)


dataset="3DHOI"
dir_path="../../ICSS"#%s" % dataset
paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))

result=diff_voting(paths["common"],paths["binary"],clf="LR")
result.report()