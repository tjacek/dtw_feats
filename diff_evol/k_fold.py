import sys
sys.path.append("..")
import numpy as np
from sklearn.metrics import log_loss,mean_squared_error,mean_absolute_error
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import ens,exp,files,learn

class KFoldGen(object):
	def __init__(self,n_splits=2):
		self.kf=KFold(n_splits=n_splits,shuffle=True)

	def __call__(self,names):
		self.kf.get_n_splits(names)
		for train_index, test_index in self.kf.split(names):
			train_names=[ names[i] for i in train_index]
			yield files.SetSelector(train_names)

class LossFunc(object):
	def __init__(self,all_votes,metric):
		self.all_votes=all_votes
		self.metric=metric

	def __call__(self,weights):
		loss=0
		for votes_i in self.all_votes:
			result_i=votes_i.weighted(weights)
			y_true_i=result_i.true_one_hot()
			y_pred_i=result_i.y_pred
			loss+=self.metric(y_true_i,y_pred_i)
		return loss

class LogLoss(LossFunc):
	def __init__(self,all_votes):
		LossFunc.__init__(self,all_votes,log_loss)
		
class MSELoss(LossFunc):
	def __init__(self,all_votes):
		LossFunc.__init__(self,all_votes,mean_squared_error)	

class LinearLoss(LossFunc):
	def __init__(self,all_votes):
		LossFunc.__init__(self,all_votes,mean_absolute_error)

def split_voting(common,deep,clf="LR"):#,n_split=10):
	datasets=ens.read_dataset(common,deep)
	train=[data_i.split()[0] for data_i in datasets]
	names=list(train[0].keys())
	kf=KFoldGen()
	all_votes=get_votes(train,clf,names,kf)
	loss_func=LinearLoss(all_votes)
	weights=optimize(loss_func,len(all_votes[0]))
	test_weights(datasets,clf,weights)

def get_votes(train,clf,names,selector_gen):
	all_votes=[]
	for selector_i in selector_gen(names):
		results=learn.train_ens(train,clf=clf,selector=selector_i)
		all_votes.append(ens.Votes(results))
	return all_votes

def test_weights(datasets,clf,weights):
	votes=ens.Votes(learn.train_ens(datasets,clf))
	result=votes.weighted(weights)
	result.report()

def optimize(loss_func,n_votes):
	starting_values = [0.5]*n_votes#len(all_votes[0])
	cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
	bounds = [(0,1)]*n_votes#len(all_votes[0])
	res = minimize(loss_func, starting_values, 
		method='SLSQP', bounds=bounds, constraints=cons)
	print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
	print('Best Weights: {weights}'.format(weights=res['x']))
	return res['x']

dataset="3DHOI"
dir_path="../../ICSS"#%s" % dataset
paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
split_voting(paths["common"],paths["binary"],clf="LR")