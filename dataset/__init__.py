import numpy as np
import dataset.instances
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

class Dataset(object):
    def __init__(self,X,y,persons,names):
        self.X=X
        self.y=y
        self.persons=persons
        self.names=names

    def __len__(self):
        return len(self.y)

    def dim(self):
        return self.X.shape[1]	
    
    def norm(self):
        self.X=preprocessing.scale(self.X)

    def select(self,n=100):
        if(self.dim()<n or n==0):
            return self
        svc = SVC(kernel='linear',C=1)
        rfe = RFE(estimator=svc,n_features_to_select=n,step=1)
        rfe.fit(self.X, self.y)
        self.X= rfe.transform(self.X)
        print("New dim: ")
        print(self.X.shape)    

    def split(self):
        insts=self.to_instances()
        train,test=insts.split()
        return to_dataset(train),to_dataset(test)

    def to_instances(self):
        n_insts=len(self)
        insts=[]
        for i in xrange(n_insts):
            x_i,y_i,person_i,name_i=self.X[i],self.y[i],self.persons[i],self.names[i]
            inst_i=dataset.instances.Instance(x_i,y_i,person_i,name_i)
            insts.append(inst_i)
        return instances.InstsGroup(insts)

def read_dataset(in_path):
    if(type(in_path)==list):
        datasets=[read_dataset(path_i) for path_i in in_path]
        return unify_datasets(datasets)
    insts=dataset.instances.from_files(in_path)
    return to_dataset(insts)

def to_dataset(insts):
    X,y,persons,names=[],[],[],[]
    insts=insts.as_dict()
    names=insts.keys()
    names.sort()
    for i,name_i in enumerate(names):
        inst_i=insts[name_i]
        X.append(np.array(inst_i.data))
        y.append(inst_i.cat)
        persons.append(inst_i.person)
    X=np.array(X)
    return Dataset(X=X,y=y,persons=persons,names=names)

def unify_datasets(datasets):
    datasets=[dataset_i 
                for dataset_i in datasets
                    if(not dataset_i is None)]
    if(len(datasets)==1):
        return datasets[0]
    feats=[ dataset_i.X for dataset_i in datasets]
    united_X=np.concatenate(feats,axis=1)
    print(united_X.shape)
    first=datasets[0]
    return Dataset(united_X,first.y,first.persons,first.names)