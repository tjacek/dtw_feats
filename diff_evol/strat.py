import sys
sys.path.append("..")
#from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import log_loss
from scipy.optimize import minimize
import random
import ens,learn,exp,files,learn

def split_voting(common,deep,clf="LR"):#,n_split=10):
	datasets=ens.read_dataset(common,deep)
	weights=find_weights(datasets)
	votes=ens.Votes(learn.train_ens(datasets,clf))
	result=votes.weighted(weights)
	result.report()

def find_weights(datasets):
	datasets=[data_i.split()[0] for data_i in datasets]
	names=list(datasets[0].keys())
#	votes=[shuffle_split(names,datasets,test_size=0.25)
#			for i in range(n_split)]
	votes=shuffle_split(names,datasets,test_size=0.25)
	def log_loss_func(weights):
		result=votes.weighted(weights)
		y_true=result.true_one_hot()
		y_pred=result.y_pred
		return log_loss(y_true,y_pred)
	starting_values = [0.5]*len(votes)
	cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
	bounds = [(0,1)]*len(votes)
	res = minimize(log_loss_func, starting_values, 
		method='SLSQP', bounds=bounds, constraints=cons)
	print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
	print('Best Weights: {weights}'.format(weights=res['x']))
	return res['x']

def shuffle_split(names,datasets,test_size=0.25):
	selector_i=custom_split(names, test_size=test_size)
	results=learn.train_ens(datasets,clf="LR",selector=selector_i)
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
split_voting(paths["common"],paths["binary"],clf="LR")