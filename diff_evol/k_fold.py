import sys
sys.path.append("..")
import numpy as np
from sklearn.metrics import log_loss
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

class LogLoss(object):
	def __init__(self,all_votes):
		self.all_votes=all_votes

	def __call__(self,weights):
		loss=0
		for votes_i in self.all_votes:
			result_i=votes_i.weighted(weights)
			y_true_i=result_i.true_one_hot()
			y_pred_i=result_i.y_pred
#			raise Exception(y_pred_i)
			loss+=log_loss(y_true_i,y_pred_i)
		return loss

def split_voting(common,deep,clf="LR"):#,n_split=10):
	datasets=ens.read_dataset(common,deep)
	datasets=[data_i.split()[0] for data_i in datasets]
	names=list(datasets[0].keys())
#	train,test=split(names)
	kf=KFoldGen()
	all_votes=[ens.Votes(learn.train_ens(datasets,clf=clf,selector=selector_i))
					for selector_i in kf(names)]

	loss_func=LogLoss(all_votes)
	starting_values = [0.5]*len(all_votes[0])
	cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
	bounds = [(0,1)]*len(all_votes[0])
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