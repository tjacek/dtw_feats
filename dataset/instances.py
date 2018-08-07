import numpy as np
import plot,utils
import seqs.select 
import sklearn.datasets
import ensemble.single_cls
from sets import Set

class InstsGroup(object):
    def __init__(self,instances):
        self.instances=instances
    
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, key):
        return self.instances[key]

    def as_dict(self):
        return { inst_i.name:inst_i for inst_i in self.instances}

    def names(self):
        return [inst_i.name for inst_i in self.instances]

    def cats(self):
        return [inst_i.cat for inst_i in self.instances] 
    
    def get_cat(self,i):
        return [inst_i.name for inst_i in self.instances
                    if(inst_i.cat==i)]

    def n_cats(self):
        cats=np.unique(self.cats())
        return cats.shape[0]

    def split(self, selector=None):
        if(selector is None):
            selector=seqs.select.ModuloSelector(n=1)
        train,test=[],[]
        for inst_i in self.instances:       
            if(selector(inst_i)):
                train.append(inst_i)
            else:
                test.append(inst_i)
        return InstsGroup(train),InstsGroup(test)

class Instance(object):
    def __init__(self,data,cat,person,name):
        self.data = data
        self.cat=int(cat)
        self.person=int(person)
        self.name=name

    def  __str__(self):
        feats=[ str(feat_i) for feat_i in list(self.data)]
        feats=",".join(feats)
        return "%s#%s#%s#%s" % (feats,self.cat,self.person,self.name)

def from_files(in_path):
    with open(in_path) as f:
         lines=f.readlines()
    insts=[ parse_instance(line_i)
                    for line_i in lines]                 
    return InstsGroup(insts)

def parse_instance(line_i):
    feats,cat,person,name=line_i.split("#")
    data=utils.str_to_vector(feats)
    return Instance(data,cat,person,name)

def empty_instance(name):
    cat,person,e=utils.extract_numbers(name)
    return Instance(None,cat,person,name)

def to_txt(out_path,insts):
    lines=[str(inst_i) for inst_i in insts]
    utils.save_string(out_path,lines)