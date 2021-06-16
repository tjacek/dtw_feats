import numpy as np,random
from sklearn.model_selection import StratifiedShuffleSplit
from distutils.dir_util import copy_tree
import feats,files,learn  

def cross_validate(feat_dict,n=10):
	acc=[validate(feat_dict) for i in range(n)]
	print(acc)
	return np.mean(acc)

def validate(feat_dict):
	if(type(feat_dict)==str):
		feat_dict=feats.read(feat_dict)[0]
	train=feat_dict.split()[0]
	cross_train=random_split(train)
	result=learn.train_model(cross_train,binary=False,clf_type="LR")
	return result.get_acc()

def person(feat_dict):
	if(type(feat_dict)==str):
		feat_dict=feats.read(feat_dict)[0]
	train=feat_dict.split()[0]
	def helper(name_i):
		return name_i.get_person()==1
	result=learn.train_model(train,binary=False,
		clf_type="LR",selector=helper)
	return result.get_acc()

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

def rename_frames(in_path,out_path,rename):
	paths=files.top_files(in_path)
	files.make_dir(out_path)
	for path_i in paths:
		name_i=path_i.split("/")[-1]
		out_i="%s/%s" % (out_path,rename[name_i])
		print(path_i)
		print(out_i)
		copy_tree(path_i,out_i)

if __name__ == "__main__":
    path="../ICSS/3DHOI/1D_CNN/feats"
    import ens
    datasets=ens.read_dataset(path,None)[0]
    random_split(datasets)