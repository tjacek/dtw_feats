import sys
sys.path.append("..")
import numpy as np
import exp,ens,feats,learn

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

    def reset(self):
        self.s_name=None

def cv_exp(common,binary,out_path=None,clf="LR"):
    datasets=ens.read_dataset(common,binary)
    validation=[ CrossVal(0.1*(i+1)) for i in range(2,10)]
    result_dict={}
    for valid_i in validation:
        votes_i= ens.Votes(valid_i(datasets,clf)[1])
        result_dict[str(valid_i)]=votes_i.voting(False)
    exp.result_exp("no weights",result_dict,out_path)

if __name__ == "__main__":
    dataset="3DHOI"
    dir_path="../../ICSS"#%s" % dataset
    paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
    paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
    cv_exp(paths["common"],paths["binary"],out_path="auc.csv",clf="LR")
#result=votes.voting()
#result.report()