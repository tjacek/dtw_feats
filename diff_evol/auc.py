import sys
sys.path.append("..")
import numpy as np
import exp,ens,feats,learn

def auc_exp(common,binary,clf="LR"):
    datasets=ens.read_dataset(common,binary)
    dataset=[date_i.norm() for date_i in datasets]
    dataset=[date_i.split() for date_i in datasets]
    dataset=list(zip(*dataset))
    train,test=dataset[0],dataset[1]
    names=subsample(train[0],p=0.9)
    results=[]
    for i,train_i in enumerate( train):
        new_train_i={name_j:train_i[name_j]
            for name_j in names}
        new_train_i=feats.Feats(new_train_i)
        model_i=learn.make_model(new_train_i,clf)
        X_test,y_true=test[i].get_X(),test[i].get_labels()
        y_pred=model_i.predict_proba(X_test)
        results.append(learn.Result(y_true,y_pred,test[i].names()))
    return ens.Votes(results)

def subsample(train,p=0.1):
    all_names=list(train.keys())
    return [name_i 
                for name_i in all_names
                    if(np.random.uniform()<p)]


dataset="3DHOI"
dir_path="../../ICSS"#%s" % dataset
paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
votes=auc_exp(paths["common"],paths["binary"],clf="LR")
result=votes.voting()
result.report()