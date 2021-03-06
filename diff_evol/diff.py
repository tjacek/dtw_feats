import sys
sys.path.append("..")
import numpy as np
from scipy.optimize import differential_evolution
import exp,ens,learn,feats,files
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
        self.iter=0

    def __call__(self,weights):
        self.iter+=1
        print(self.iter)
        weights=weights/np.sum(weights)
        result=self.all_votes.weighted(weights)
        return mse_fun(result)

class DiffEvol(object):
    def __init__(self,maxiter=3,popsize=5,polish=False):
        self.maxiter=maxiter
        self.popsize=popsize
        self.polish=polish

    def __call__(self,loss_fun,bound_w):
        return differential_evolution(loss_fun, bound_w, 
                tol=1e-7,maxiter=self.maxiter,
                popsize=self.popsize,polish=self.polish)

def mse_fun(result):
    y_true=result.true_one_hot()
    y_pred=result.y_pred
    squared_mean=[np.sum((true_i- pred_i)**2)
                for true_i,pred_i in zip(y_true,y_pred)]
    return  np.mean(squared_mean)

class OptimWeights(object):
    def __init__(self, loss=None,validation=None,
                    popsize=1):#,maxiter=1):
        if(loss is None):
            loss=Corl
        self.loss = loss
        if(validation is None):
            validation=validation_votes	
        self.validation=validation
        self.popsize=popsize
#        self.maxiter=maxiter

    def __call__(self,paths,clf="LR"):
        datasets=ens.read_dataset(paths['common'],paths['binary'])   
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
        return result,weights

    def find_weights(self,results,clf="LR"):
        loss_fun=self.loss(results)
        bound_w = [(0.01, 1.0)  for _ in results]
#        init_matrix=np.ones((6, len(bound_w)))
        result = differential_evolution(loss_fun, bound_w, 
                tol=1e-7,maxiter=3,popsize=5,polish=True)#False)
#                init=init_matrix)
#        loss_fun.iter=0
        weights=result['x']
        return weights

#def validation_votes(datasets,clf="LR"):
#    train=[data_i.split()[0] for data_i in datasets]
#    names=list(train[0].keys())
#    selector_gen=k_fold.StratGen(1)
#    selector=list(selector_gen(names))[0]
#    results= learn.train_ens(train,clf=clf,selector=selector)
#    return datasets,results

def auc_exp(paths,dir_name="auc",mediana=True,n=10):
    files.make_dir(dir_name)
    loss_dict={"MSE":MSE,"gasen":gasen.Corl,"Comb":Comb}
    for loss_name,loss_i in loss_dict.items():       
        validation=[ auc.CrossVal(0.1*(i+1)) for i in range(2,n)]
        if(mediana):
            validation=[ auc.MedianaVal(val_i,k=10) for val_i in validation]
        optim=OptimWeights(loss_i,validation)
        result_dict=optim(paths)
        out_i="%s/%s.csv" % (dir_name,loss_name)
        result_dict=weight_desc(result_dict,eps=0.02)    
        exp.result_exp(loss_name,result_dict,out_i)

def weight_desc(result_dict,eps=0.02):
    weight_dict={}
    for name_i,pair_i in result_dict.items():
        result_i,weights_i=pair_i
        s_clf=(weights_i>eps)
        n_clf=weights_i[s_clf].shape[0]
        s_clf=str(np.where(s_clf==True)[0])
        new_name_i="%s,%d,%s" % (name_i,n_clf,s_clf)
        weight_dict[new_name_i]=result_i
    return weight_dict

def single_exp(paths,loss_type,out_path,p=0.5,k=10):
    loss_dict={"MSE":MSE,"gasen":gasen.Corl,"Comb":Comb}
    valid=auc.CrossVal(p)
    if(p<1.0):
        valid=auc.MedianaVal(valid,k=k)
    optim=OptimWeights(loss_dict[loss_type],valid)
    result=optim(paths)[0]
    result.report()
    result.get_cf(out_path)

if __name__ == "__main__":
    dataset="MHAD"
    dir_path="../../ICSS"#%s" % dataset
    paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
    paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
    print(paths)
#    optim=auc_exp(paths,"MHAD")
    single_exp(paths,"Comb","cf/%s" % dataset,p=1.0)