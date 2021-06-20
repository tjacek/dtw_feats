import numpy as np,random
from sklearn.model_selection import StratifiedShuffleSplit
from random import getrandbits
#from distutils.dir_util import copy_tree
import json
from collections import defaultdict
import feats,files,learn,exp,ens


def save_rename(id,rename):
    json.dump(rename,open("%s.json" % id,'w'))

def read_rename(id):
    rename= json.load(open("%s.json" % id))
    return { files.Name(name_i):files.Name(rename_i) 
                for name_i,rename_i in rename.items()}

def random_cat(feat_dict):
    if(type(feat_dict)==str):
    	feat_dict=feats.read(feat_dict)[0]
    by_cat=defaultdict(lambda :[])
    for name_i in feat_dict.keys():
        by_cat[name_i.get_cat()].append(name_i)
    rename={}
    for cat_i,names_i in by_cat.items(): 
        random.shuffle(names_i)
        for j,name_j in enumerate(names_i):
            new_name_j="%d_%d_%d" % (name_j.get_cat()+1,j%2,j)
            rename[name_j]=new_name_j
    return rename

#def person(feat_dict):
#	if(type(feat_dict)==str):
#		feat_dict=feats.read(feat_dict)[0]
#	train=feat_dict.split()[0]
#	def helper(name_i):
#		return name_i.get_person()==1
#	result=learn.train_model(train,binary=False,
#		clf_type="LR",selector=helper)
#	return result.get_acc()

def random_split(train):
    sss=StratifiedShuffleSplit(n_splits=1, 
            test_size=0.5, random_state=0)
    X,y,names=train.as_dataset()
    a=sss.split(X,y)
    new_dict=feats.Feats()
    for train_index, test_index in a:
        for i in test_index:
        	name_i="%d_0_%d" % (names[i].get_cat(),i)
        	new_dict[names[i]]=train[names[i]]
        for i in train_index:
        	name_i="%d_1_%d" % (names[i].get_cat(),i)
        	new_dict[names[i]]=train[names[i]]
    return new_dict

def rename_frames(paths,json_path):
    datasets=ens.read_dataset(paths["common"],paths["binary"])
    helper=get_renam_fun(json_path)
    if(get_fun):
        return helper
    new_datasets=datasets #[helper(data_i) for data_i in datasets]
    votes=ens.Votes(learn.train_ens(new_datasets,clf="LR"))
    result_i=votes.voting(False)
    result_i.report()

def get_renam_fun(json_path):
    rename=read_rename(json_path)
    def helper(data_i):
        feat_i=feats.Feats()
        for name_i,rename_i in rename.items():
            print((rename_i,name_i))
            feat_i[rename_i]=data_i[name_i]
        return feat_i
    return helper

if __name__ == "__main__":  
    dataset="3DHOI"
    dir_path=".."
    paths=exp.basic_paths(dataset,dir_path,"dtw",None)
    paths["common"]=["../3DHOI/1D_CNN/feats"]
    rename_frames(paths,"rename")
#    rename=random_cat("../3DHOI/1D_CNN/feats")
#    save_rename("rename",rename)