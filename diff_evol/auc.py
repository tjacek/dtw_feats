import sys
sys.path.append("..")
import numpy as np
import exp,ens,feats,learn

def auc_exp(common,binary,clf="LR"):
    datasets=ens.read_dataset(common,binary)
    dataset=[date_i.split() for date_i in datasets]
    dataset=list(zip(*dataset))
    train,test=dataset[0],dataset[1]
    names=subsample(train[0],p=0.9)
    models=[]
    for train_i in train:
        new_train_i={name_j:train_i[name_j]
            for name_j in names}
        new_train_i=feats.Feats(new_train_i)
        models.append(learn.make_model(new_train_i,clf))
#    print(dataset[0].keys())

def subsample(train,p=0.1):
    all_names=list(train.keys())
    return [name_i 
                for name_i in all_names
                    if(np.random.uniform()<p)]


dataset="3DHOI"
dir_path="../../ICSS"#%s" % dataset
paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
auc_exp(paths["common"],paths["binary"],clf="LR")