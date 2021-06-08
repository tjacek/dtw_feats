import sys
sys.path.append("..")
import numpy as np
from scipy.optimize import differential_evolution
import exp,ens,k_fold,learn,feats,files

class Corl(object):
    def __init__(self,all_votes):
        self.all_votes=ens.Votes(all_votes)	
        self.d=[ result_i.true_one_hot() 
                  for result_i in all_votes]
    
    def __call__(self,weights):
        weights=weights/np.sum(weights)
        C=self.corl()
        n_clf=len(self.all_votes)
        loss=0
        for i in range(n_clf):
            for j in range(n_clf):
                loss+=weights[i]*weights[j] * C[i,j] 	
        return 1.0*loss

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

class CrossVal(object):
    def __init__(self,p=0.3):
        self.p=p
        self.s_name=None

    def __call__(self,datasets,clf="LR"):
        results=[]
        new_datasets=[]
        for data_i in datasets:
            train_i,test_i=data_i.split()
            s_train_i=self.subsample(train_i)
            s_data_i={**s_train_i,**test_i}
            s_data_i= feats.Feats(s_data_i)
            new_datasets.append(s_data_i)
            result_i=learn.train_model(s_data_i,
            	binary=False,clf_type=clf)
            results.append(result_i)
        return new_datasets,results

    def subsample(self,train):
        all_names=list(train.keys())
        if(self.s_name is None):
            self.s_name=[name_i 
                    for name_i in all_names
                        if(np.random.uniform()<self.p)]
        s_train={name_i:train[name_i] 
                    for name_i in self.s_name}
        return feats.Feats(s_train)

    def __str__(self):
        return str(self.p)

def auc_exp(paths,n=10):
    files.make_dir("auc")
    loss_dict={"MSE":MSE ,"gasen":Corl}
    for loss_name,loss_i in loss_dict.items(): 
        validation=[ CrossVal(0.1*(i+1)) for i in range(2,n)]
        optim=OptimWeights(loss_i,validation)
        result_dict=optim(paths["common"],paths["binary"])
        out_i="auc/%s.csv" % loss_name
        exp.result_exp(loss_name,result_dict,out_i)

dataset="3DHOI"
dir_path="../../ICSS"#%s" % dataset
paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
print(paths)
optim=auc_exp(paths) #OptimWeights(Corl,CrossVal())