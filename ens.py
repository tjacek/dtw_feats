import numpy as np
import learn,feats
#import __learn as learn
#import __feats as feats

def read_dataset(common_path,deep_path):
    if(not common_path):
        return feats.read(deep_path)
    if(not deep_path):
        return feats.read(common_path)
    common_data=feats.read(common_path)[0]
    deep_data=feats.read(deep_path)
    datasets=[common_data+ data_i 
                for data_i in deep_data]
    return datasets

def voting(results,binary=True):
    votes=np.array([ result_i.as_numpy() 
                for result_i in results])
    votes=np.sum(votes,axis=0)
    return learn.Result(results[0].y_true,votes,results[0].names)

def ensemble(common_path,binary_path,binary=True,clf="SVC"):
    datasets=read_dataset(common_path,binary_path)
    results=[learn.train_model(data_i,clf_type=clf,binary=binary)
                for data_i in datasets]
    result=voting(results)
    print(result.get_acc()) 
    return result

def simple_exp(common_path,deep_path,clf="SVC"):
	dataset=read_dataset(common_path,deep_path)[0]
	result=learn.train_model(dataset,clf_type=clf)
	print(result.get_acc())

if __name__ == "__main__":
#common_path=["old2/agum/basic/feats","old2/simple/basic/feats"]
    common_path="good/ae_basic"
    binary_path="good/ens"
    result=ensemble(common_path,binary_path,clf="LR")
    result.save("out")