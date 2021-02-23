import sys
sys.path.append("..")
import numpy as np
import acc,ens,learn,files,selection

def total_person_selection(common_path,binary_path,n,n_clf,clf="LR"):
    def helper(common_path,binary_path,clf="LR"):
        datasets=ens.read_dataset(common_path,binary_path)
        results=[ person_acc(data_i) for data_i in datasets]
        return ens.Votes(results)
    selection.random_selection(common_path,binary_path,n,n_clf,clf,helper)

def person_selection(common_path,binary_path,clf="LR"):
    datasets=ens.read_dataset(common_path,binary_path)
    clf_acc=np.array([ person_acc(data_i).get_acc()
                        for data_i in datasets])    
    s_clf=acc.dataset_selection(datasets,clf_acc)
    print(len(s_clf))
    return ens.ensemble(common_path,binary_path,True,clf,s_clf)[0]

def person_acc(data_i):
    train=data_i.split()[0]
    persons= set([name_i.get_person() for name_i in train.keys()])
    results=[]
    print("person acc")
    for j in list(persons):
        def helper(name_i):
            cat_i=name_i.get_cat()+1
            person_i=int(name_i.get_person()!=j)
            return "%d_%d" % (cat_i,person_i)
        rename_j={ name_i:"%s_%d" % (helper(name_i),i) 
                    for i,name_i in enumerate(train.keys())}
        train_j=train.rename(rename_j)
        result_j=learn.train_model(train_j,binary=False,clf_type="LR")
        results.append(result_j)
    return unify_results(results)

def unify_results(results):
    y_true=[result_i.y_true for result_i in results]
    y_pred=[result_i.y_pred for result_i in results]
    names=[result_i.names for result_i in results]
    raw=[ files.flatten(x) for x in [y_true,y_pred,names]]
    return learn.Result(raw[0],raw[1],raw[2])

if __name__ == "__main__":
    dataset="../../dtw_paper/MSR"
    common="%s/common/MSR_500" % dataset
    binary="%s/binary/stats/feats" %dataset
#    result=person_selection(common,binary,clf="LR")
#    result.report()
    total_person_selection(common,binary,1000,20,clf="LR")