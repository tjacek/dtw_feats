import sys
sys.path.append("..")
import numpy as np
from scipy.optimize import differential_evolution
import exp,ens,learn,feats,files,
import k_fold,auc,gasen

class Comb(object):
    def __init__(self,all_votes):
        self.corl=gasen.Corl(all_votes)
        self.mse=MSE(all_votes)

    def __call__(self,weights):
        return self.corl(weights)+self.mse(weights) 	

class MSE(object):
    def __init__(self,all_votes):
        self.all_votes=ens.Votes(all_votes)

    def __call__(self,weights):
        weights=weights/np.sum(weights)
        result=self.all_votes.weighted(weights)
        y_true=result.true_one_hot()
        y_pred=result.y_pred
        squared_mean=[np.sum((true_i- pred_i)**2)
                for true_i,pred_i in zip(y_true,y_pred)]
        return np.mean(squared_mean)

class OptimWeights(object):
    def __init__(self, loss=None,validation=None):
        if(loss is None):
            loss=Corl
        self.loss = loss
        if(validation is None):
            validation=validation_votes	
        self.validation=validation

    def __call__(self,common,deep,clf="LR"):
        datasets=ens.read_dataset(common,deep)   
        def helper(valid):
            print("Valid")	
            new_datasets,results=valid(datasets)
            return self.single_optim(new_datasets,results,clf)
        if(type(self.validation)==list ):
            results={str(valid_i):helper(valid_i)
                        for valid_i in self.validation}
            return results
        else:
            return helper(self.validation)

    def single_optim(self,datasets,results,clf):
        weights=self.find_weights(results)
        results=learn.train_ens(datasets,clf)
        votes=ens.Votes(results)
        result=votes.weighted(weights)
        return result

    def find_weights(self,results,clf="LR"):
        loss_fun=self.loss(results)
        bound_w = [(0.01, 1.0)  for _ in results]
        result = differential_evolution(loss_fun, bound_w, 
    			maxiter=10, tol=1e-7)
        weights=result['x']
        return weights

def validation_votes(datasets,clf="LR"):
    train=[data_i.split()[0] for data_i in datasets]
    names=list(train[0].keys())
    selector_gen=k_fold.StratGen(1)
    selector=list(selector_gen(names))[0]
    results= learn.train_ens(train,clf=clf,selector=selector)
    return datasets,results

def auc_exp(paths,dir_name="auc"):
    files.make_dir(dir_name)
    loss_dict={"MSE":MSE,"Comb":Comb,"gasen":gasen.Corl}
    for loss_name,loss_i in loss_dict.items():       
        validation=[ auc.CrossVal(0.1*(i+1)) for i in range(2,10)]
        validation=[ auc.MedianaVal(val_i) for val_i in validation]
        optim=OptimWeights(loss_i,validation)
        result_dict=optim(paths["common"],paths["binary"])
        out_i="%s/%s.csv" % (dir_name,loss_name)
        exp.result_exp(loss_name,result_dict,out_i)

if __name__ == "__main__":
    dataset="3DHOI"
    dir_path="../../ICSS"#%s" % dataset
    paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
    paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
    print(paths)
    optim=auc_exp(paths,"auc")