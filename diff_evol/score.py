import sys
sys.path.append("..")    
import numpy as np
import ens,auc,exp

class ScoreLoss(object):
    def __init__(self,results):
        self.results=results

    def __call__(self,score_weights):
        score_weights=score_weights/np.sum(score_weights)

def score_opt(paths,clf="LR"):
    datasets=ens.read_dataset(paths['common'],paths['binary'])
    val=auc.CrossVal(0.5)
    new_datasets,results=val(datasets,clf)
    score_weights=borda_weights(len(datasets))
    loss=ScoreLoss(results)
    loss(score_weights)

def borda_weights(n_cats):
    weights=[float(i) for i in range(n_cats)]
    weights.reverse()
    return np.array( weights)

if __name__ == "__main__":
    dataset="3DHOI"
    dir_path="../../ICSS"#%s" % dataset
    paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
    paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
    score_opt(paths,clf="LR")