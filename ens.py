import numpy as np
import pickle
import learn,feats
#import __learn as learn
#import __feats as feats

class Votes(object):
    def __init__(self,results):
        self.results=results

    def __len__(self):
        return len(self.results)

    def voting(self,binary=False):
        if(binary):
            votes=np.array([ result_i.as_hard_votes() 
                    for result_i in self.results])
        else:
            votes=np.array([ result_i.as_numpy() 
                    for result_i in self.results])
        votes=np.sum(votes,axis=0)
        return learn.Result(self.results[0].y_true,votes,self.results[0].names)

    def indv_acc(self):
        return [ result_i.get_acc() for result_i in self.results]
    
    def save(self,out_path):
        with open(out_path, 'wb') as out_file:
            pickle.dump(self, out_file)

def make_votes(common_path,binary_path,clf="SVC"):
    datasets=read_dataset(common_path,binary_path)
    if(len(datasets)==0):
        raise Exception("No data at %s" % binary_path)
    results=[learn.train_model(data_i,clf_type=clf,binary=False)
                for data_i in datasets]
    return Votes(results)

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

def ensemble(common_path,binary_path,binary=True,clf="SVC"):
    votes=make_votes(common_path,binary_path,clf)
    print(votes.indv_acc())
    result=votes.voting(binary)
    print(result.get_acc()) 
    return result

def simple_exp(common_path,binary_path,clf="SVC"):
	dataset=read_dataset(common_path,binary_path)[0]
	result=learn.train_model(dataset,clf_type=clf)
	print(result.get_acc())

def show_acc(common_path,binary_path,binary=True,clf="SVC"):
    datasets=read_dataset(common_path,binary_path)
    results=[learn.train_model(data_i,clf_type=clf,binary=binary)
                for data_i in datasets]
    return [ result_i.get_acc() for result_i in results]

if __name__ == "__main__":
    deep=['../ICSS_exp/3DHOI/common/stats/feats']
    binary='../ICSS_exp/3DHOI/ens/lstm/feats'
    dtw=['../ICSS_exp/3DHOI/dtw/corl/dtw', '../ICSS_exp/MHAD/dtw/max_z/dtw']
    result=ensemble(dtw+deep,binary,clf="LR",binary=False)