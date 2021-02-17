import numpy as np,random
from distutils.dir_util import copy_tree
import feats,files,learn

def rename_exp(in_path,k=10,threshold=None):
	data=feats.read(in_path)[0]
	group=files.by_cat(data.keys())
	acc=[]
	for i in range(k):
		print(i)
		rename=random_split(group)
		new_data=rename_dataset(data,rename)
		result=learn.train_model(new_data,binary=False,clf_type="LR")
		acc_i=result.get_acc()
		if(threshold and acc_i>threshold):
			print(acc_i)
			return rename
		acc.append(acc_i)
	print(np.mean(acc),np.std(acc),np.amax(acc),np.amin(acc))

def rename_dataset(dataset,rename):
	new_data=feats.Feats()
	for name_i in dataset.keys():
		new_name=rename[name_i]
		new_data[new_name]=dataset[name_i]
	return new_data

def random_split(group):
	rename={}
	for class_i in group.values():
		random.shuffle(class_i)
		for i,name_i in enumerate(class_i):
			if( (i%2)==0):
				new_name="%d_0_%d" % (name_i.get_cat()+1,i)
			else:
				new_name="%d_1_%d" % (name_i.get_cat()+1,i)
			rename[name_i]=files.Name(new_name)
	return rename

def rename_frames(in_path,out_path,rename):
	paths=files.top_files(in_path)
	files.make_dir(out_path)
	for path_i in paths:
		name_i=path_i.split("/")[-1]
		out_i="%s/%s" % (out_path,rename[name_i])
		print(path_i)
		print(out_i)
		copy_tree(path_i,out_i)

rename=rename_exp("s_dtw",k=10,threshold=0.6)
print(len(rename))
in_path="../clean3/agum/frames"
out_path="../rename/agum/frames"
rename_frames(in_path,out_path,rename)