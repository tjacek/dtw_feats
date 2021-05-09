import sys
sys.path.append("..")
#from sklearn.model_selection import StratifiedShuffleSplit
import random
import ens,learn,exp,files

def diff_voting(common,deep,clf="LR",n_split=10):
	datasets=ens.read_dataset(common,deep)
	datasets=[data_i.split()[0] for data_i in datasets]
	names=list(datasets[0].keys())
	votes=[shuffle_split(names,datasets,test_size=0.25)
			for i in range(n_split)]
	acc=[ vote_i.voting(False).get_acc() for vote_i in votes]
	print(acc)

def shuffle_split(names,datasets,test_size=0.25):
	selector_i=custom_split(names, test_size=test_size)
	results=[learn.train_model(data_i,clf_type="LR",selector=selector_i)
				for data_i in datasets]
	return ens.Votes(results)

class SetSelector(object):
	def __init__(self,names):
		self.train=set(names)

	def __call__(self,name_i):
		return name_i in self.train

def custom_split(names, test_size=0.25):
	split_size= int(len(names)* (1.0-test_size))
	random.shuffle(names)
	split=names[:split_size]
	return SetSelector(split)

dataset="3DHOI"
dir_path="../../ICSS"#%s" % dataset
paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
diff_voting(paths["common"],paths["binary"],clf="LR")