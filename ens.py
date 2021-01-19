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

    def weighted(self,weights):
        votes=np.array([ weight_i*result_i.as_numpy() 
                    for weight_i,result_i in zip(weights,self.results)])
        votes=np.sum(votes,axis=0)
        return learn.Result(self.results[0].y_true,votes,self.results[0].names)

    def indv_acc(self):
        return [ result_i.get_acc() for result_i in self.results]
    
    def acc_matrix(self):
        n_cats=self.results[0].n_cats()
        n_clf=len(self)
        acc=[[ self.results[i].cat_acc(j)
                for i in range(n_clf)]
                    for j in range(n_cats)]
        return np.array(acc)

    def save(self,out_path):
        with open(out_path, 'wb') as out_file:
            pickle.dump(self, out_file)

def make_votes(common_path,binary_path,clf="LR"):
    datasets=read_dataset(common_path,binary_path)
    if(len(datasets)==0):
        raise Exception("No data at %s" % binary_path)
    results=[learn.train_model(data_i,clf_type=clf,binary=False)
                for data_i in datasets]
    return Votes(results)

def read_dataset(common_path,deep_path):
    if(not common_path):
        return read_deep(deep_path)#feats.read(deep_path)
    if(not deep_path):
        return feats.read(common_path)
    common_data=feats.read(common_path)[0]
    deep_data=read_deep(deep_path)
    datasets=[common_data+ data_i 
                for data_i in deep_data]
    return datasets

def read_deep(deep_path):
    if(type(deep_path)==list):
        datasets=[]
        for deep_i in deep_path:
            datasets+=feats.read(deep_i)
        return datasets
    return feats.read(deep_path)

def ensemble(common_path,binary_path,binary=True,clf="SVC"):
    votes=make_votes(common_path,binary_path,clf)
    result=votes.voting(binary)
    print(result.get_acc()) 
    return result

if __name__ == "__main__":
    dataset="MHAD"
    path='../ICSS_exp/%s/' % dataset
    deep=['%s/common/1D_CNN/feats' % path]
    binary='%s/ens/lstm_gen/feats' % path
#    binary='../ICSS_sim/%s/sim/feats' % dataset
#    binary=['%s/ens/lstm_gen/feats' % path,'../ICSS_sim/%s/sim/feats' % dataset]
#    dtw=['%s/dtw/corl/dtw' % path, '%s/dtw/max_z/dtw' % path]
    dtw="handcrafted/3DHOI/max_z/out"
    result=ensemble(dtw,None,clf="LR",binary=False)
    result.report()