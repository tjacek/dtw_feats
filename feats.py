import os.path
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
import files#,tools

class FeatureSet(object):
    def __init__(self,X,info):
        self.X=X
        self.info=info

    def __add__(self,feat_i):
        if(len(self)==len(feat_i)):
            new_X=np.concatenate([self.X,feat_i.X],axis=1)
            return FeatureSet(new_X,self.info)
        new_info=self.common_names(feat_i)
        new_info.sort()
        feat1,feat2=self.to_dict(),feat_i.to_dict()
        feat1={ name_i:feat1[name_i] for name_i in new_info}
        feat2={ name_i:feat2[name_i] for name_i in new_info}
        new_X=np.concatenate([from_dict(feat1).X,from_dict(feat2).X],axis=1)
        return FeatureSet(new_X,new_info)

    def __len__(self):
        return len(self.info)

    def dim(self):
        return self.X.shape[1]

    def n_cats(self):
        return len(set(self.get_labels()))

    def get_labels(self):
        return [ int(info_i.split('_')[0])-1 for info_i in self.info]

    def to_dict(self):
        return { self.info[i]:x_i 
                    for i,x_i in enumerate(self.X)}

    def split(self,selector=None):
        feat_dict=self.to_dict()
#        train,test=tools.split(feat_dict,selector)
        train,test=files.split(feat_dict)
        return from_dict(dict(train)), from_dict(dict(test))

    def norm(self):
        self.X=preprocessing.scale(self.X)
        return self
    
    def binary(self,cat_i):
        y=self.get_labels()
        binary_y=[ "%d_%d"%(int(y_i==cat_i),i) 
                        for i,y_i in enumerate(y)]
        return FeatureSet(self.X,binary_y)

    def mutual(self):
        return mutual_info_classif(self.X,self.get_labels())

    def modify(self,new_X):
        return FeatureSet(new_X,self.info)

    def common_names(self,feat1):
        return list(set(self.info).intersection(set(feat1.info)))

    def save(self,out_path,decimals=10):
        lines=[ np.array2string(x_i,separator=",",precision=decimals) for x_i in self.X]
        lines=[ line_i.replace('\n',"")+'#'+info_i 
                    for line_i,info_i in zip(lines,self.info)]
        feat_txt='\n'.join(lines)
        feat_txt=feat_txt.replace('[','').replace(']','')
        file_str = open(out_path,'w')
        file_str.write(feat_txt)
        file_str.close()

    def by_cat(self):
        cats=[[]for i in range(self.n_cats())]
        for info_i in self.info:
            cat_i=int(info_i.split('_')[0])-1
            cats[cat_i].append(info_i)
        return cats

def read(in_path):
    if(type(in_path)==list):
        return [read_unified(in_path)]
    if(not os.path.isdir(in_path)):
        return [from_dict(read_single(in_path))]
    return [from_dict(read_single(path_i)) 
                for path_i in files.top_files(in_path)]

def from_dict(feat_dict):
    info=files.natural_sort(feat_dict.keys())
    X=np.array([feat_dict[info_i] 
                    for info_i in info])
    return FeatureSet(X,info)

def read_single(in_path,as_dict=True):
    lines=open(in_path,'r').readlines()
    feat_dict={}
    for line_i in lines:
        raw=line_i.split('#')
        if(len(raw)>1):
            data_i,info_i=raw[0],raw[-1]
            info_i= files.clean_str(info_i)
            feat_dict[info_i]=np.fromstring(data_i,sep=',')
    if(as_dict):
        return feat_dict
    return from_dict(feat_dict)

def read_unified(paths):
    datasets=[from_dict(read_single(path_i)) 
                for path_i in paths]
    full_data=datasets[0]
    for data_i in datasets[1:]:
        full_data+=data_i
    return full_data