import sys
sys.path.append("..")
import numpy as np
from sklearn.model_selection import KFold
import ens,exp,files

class KFoldGen(object):
	def __init__(self,n_splits=2):
		self.kf=KFold(n_splits=n_splits)

	def __call__(self,names):
		self.kf.get_n_splits(names)
		for train_index, test_index in self.kf.split(names):
			train_names=[ names[i] for i in train_index]
			yield train_names

def split_voting(common,deep,clf="LR"):#,n_split=10):
	datasets=ens.read_dataset(common,deep)
	names=list(datasets[0].keys())
	train,test=split(names)
	kf=KFoldGen()
	for train_i in list(kf(train)):
		print(len(train_i))
#	kf = KFold(n_splits=2)
#	kf.get_n_splits(names)
#	for train_index, test_index in kf.split(names):
#		print("TRAIN:", train_index, "TEST:", test_index)

def split(names):
	train,test=[],[]
	for name_i in names:
		if(files.person_selector(name_i)):
			train.append(name_i)
		else:
			test.append(name_i)
	return train,test



dataset="3DHOI"
dir_path="../../ICSS"#%s" % dataset
paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
split_voting(paths["common"],paths["binary"],clf="LR")