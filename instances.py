import numpy as np
import plot,utils

class Instance(object):
    def __init__(self,data,cat,person,name):
        self.data = data
        self.cat=cat
        self.person=person
        self.name=name

    def  __str__(self):
        feats=[ str(feat_i) for feat_i in list(self.data)]
        feats=",".join(feats)
        return "%s#%s#%s#%s" % (feats,self.cat,self.person,self.name)

def to_dataset(instances):
    X,y=[],[]
    for inst_i in instances:
        X.append(np.array(inst_i.data))
        y.append(inst_i.cat)	
    X=np.array(X)
    return X,y

def from_files(in_path):
    with open(in_path) as f:
         lines=f.readlines()
    return [ parse_instance(line_i)
                for line_i in lines]     

def parse_instance(line_i):
    feats,cat,person,name=line_i.split("#")
    data=utils.str_to_vector(feats)
    return Instance(data,cat,person,name)

if __name__ == "__main__":
    action_path="mra/datasets/nn19"
    insts=from_files(action_path)
    X,y=to_dataset(insts)
    plot.plot_embedding(X,y,highlist=[20])