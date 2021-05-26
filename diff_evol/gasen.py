import sys
sys.path.append("..")
import numpy as np
import ens,exp,files,learn,k_fold

def diff_voting(common,deep,clf="LR"):#,loss="mse"):
    datasets=ens.read_dataset(common,deep)
#	weights=np.ones((len(datasets),))
    weights=find_weights(datasets)#,loss=loss)
    results=learn.train_ens(datasets,clf)
    votes=ens.Votes(results)
    result=votes.weighted(weights)
    return result

def find_weights(datasets,clf="LR"):
    results=validation_votes(datasets,clf)
    raise Exception(len(results))

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