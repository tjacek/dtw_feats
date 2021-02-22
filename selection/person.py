import sys
sys.path.append("..")
import numpy as np
import acc,ens,learn

def person_selection(common_path,binary_path,clf="LR"):
    datasets=ens.read_dataset(common_path,binary_path)
    s_clf=acc.dataset_selection(datasets,person_acc)
    print(len(s_clf))
    return ens.ensemble(common_path,binary_path,True,clf,s_clf)[0]

def person_acc(data_i):
    train=data_i.split()[0]
    persons= set([name_i.get_person() for name_i in train.keys()])
    acc=[]
    for j in list(persons):
        def helper(name_i):
            cat_i=name_i.get_cat()+1
            person_i=int(name_i.get_person()!=j)
            return "%d_%d" % (cat_i,person_i)
        rename_j={ name_i:"%s_%d" % (helper(name_i),i) 
                    for i,name_i in enumerate(train.keys())}
        train_j=train.rename(rename_j)
        result_j=learn.train_model(train_j,binary=False,clf_type="LR")
        acc.append(result_j.get_acc())
    print(acc)
    return np.mean(acc)

if __name__ == "__main__":
    dataset="../../dtw_paper/MSR"
    common="%s/common/MSR_500" % dataset
    binary="%s/binary/stats/feats" %dataset
    result=person_selection(common,binary,clf="LR")
    result.report()