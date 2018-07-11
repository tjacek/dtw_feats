import numpy as np
import dataset.instances

class Dataset(object):
    def __init__(self,X,y,persons,names):
        self.X=X
        self.y=y
        self.persons=persons
        self.names=names

    def __len__(self):
        return len(self.y)	
    
    def split(self):
        insts=self.to_instances()
        train,test=instances.split_instances(insts)
        return to_dataset(train),to_dataset(test)

    def to_instances(self):
        n_insts=len(self)
        insts=[]
        for i in xrange(n_insts):
            x_i,y_i,person_i,name_i=self.X[i],self.y[i],self.persons[i],self.names[i]
            inst_i=dataset.instances.Instance(x_i,y_i,person_i,name_i)
            insts.append(inst_i)
        return insts

def to_dataset(insts):
    X,y,persons,names=[],[],[],[]
    if(type(insts)==list):
        insts={inst_i.name:inst_i for inst_i in insts}
    names=insts.keys()
    names.sort()
    for i,name_i in enumerate(names):
        inst_i=insts[name_i]
        X.append(np.array(inst_i.data))
        y.append(inst_i.cat)
        persons.append(inst_i.person)
    X=np.array(X)
    return Dataset(X=X,y=y,persons=persons,names=names)