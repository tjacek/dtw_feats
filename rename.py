import numpy as np,random
from distutils.dir_util import copy_tree
import feats,files,learn  

def cross_validate(feat_dict,n=10):
	acc=[person(feat_dict) for i in range(n)]
#	print(acc)
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
	group=files.by_cat(train.keys())
	new_dict=feats.Feats()
	for class_i in group.values():
		random.shuffle(class_i)
		for i,name_i in enumerate(class_i):
			new_name="%d_%d_%d" % (name_i.get_cat()+1,i%2,i)
			new_name=files.Name(new_name)
			new_dict[new_name]=train[name_i]
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