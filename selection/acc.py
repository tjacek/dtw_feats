import sys
sys.path.append("..")
import numpy as np
import ens,learn,feats,files

def basic_selection(common_path,binary_path,clf="LR"):
    datasets=ens.read_dataset(common_path,binary_path)
    s_clf=dataset_selection(datasets,person_acc)
    print(len(s_clf))
    return ens.ensemble(common_path,binary_path,True,clf,s_clf)[0]

def dataset_selection(datasets,aprox_acc):
    acc=np.array([ aprox_acc(data_i) for data_i in datasets])
    acc=np.array(acc)
    if(np.std(acc)==0):
    	return datasets
    acc= (acc-np.mean(acc))/np.std(acc)
    print(acc)
    s_datasets=[i for i,data_i in enumerate(datasets)
    			if(acc[i]>0)]
    return s_datasets

def validate_acc(data_i):
    train=data_i.split()[0]
    rename_i={name_i:"%d_%d_%d" % (name_i.get_cat()+1,i%2,i) 
                for i,name_i in enumerate(train.keys())}
    new_data=train.rename(rename_i)
    result=learn.train_model(new_data,binary=False,clf_type="LR")
    return result.get_acc()

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

dataset="../../dtw_paper/MSR"
common="%s/common/MSR_500" % dataset
binary="%s/binary/stats/feats" %dataset
result=basic_selection(common,binary,clf="LR")
result.report()