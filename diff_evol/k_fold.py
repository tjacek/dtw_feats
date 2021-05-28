import sys
sys.path.append("..")
import numpy as np
from sklearn.metrics import log_loss,mean_squared_error,mean_absolute_error
from scipy.optimize import minimize
from sklearn.model_selection import KFold,StratifiedShuffleSplit
import ens,exp,files,learn

class KFoldGen(object):
	def __init__(self,n_splits=2):
		self.kf=KFold(n_splits=n_splits,shuffle=True)

	def __call__(self,names):
		self.kf.get_n_splits(names)
		for train_index, test_index in self.kf.split(names):
			train_names=[ names[i] for i in train_index]
			yield files.SetSelector(train_names)

class StratGen(object):
	def __init__(self, n_split=2,test_size=0.5):
		self.sss=StratifiedShuffleSplit(n_splits=n_split, 
			test_size=test_size, random_state=0)

	def __call__(self,names):
		self.sss.get_n_splits(names)
		y=[name_i.get_cat() for name_i in names]
		for train_index, test_index in self.sss.split(names,y):
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

def split_exp(common,binary,clf="LR",out_path=None):
	datasets=ens.read_dataset(common,binary)
	gens={ "KFold,2":KFoldGen(2),"KFold,5":KFoldGen(5),
			"Strat,2":StratGen(2),"Strat,5":StratGen(5)}
	loss={"LogLoss":LogLoss,
			  "MSELoss":MSELoss,
			  "LinearLoss":LinearLoss}
	lines=exp_template(datasets,clf,gens,loss)
	print(lines)

def auc_exp(common,binary,clf="LR",out_path=None):
    datasets=ens.read_dataset(common,binary)
    gens={"Strat,0.1":StratGen(2,0.1),"Strat,0.3":StratGen(2,0.3),
           "Strat,0.5":StratGen(2,0.5),"Strat,0.9":StratGen(2,0.9)}
    loss={"MSELoss":MSELoss}
    lines=exp_template(datasets,clf,gens,loss,out_path)
    print(lines)

def exp_template(datasets,clf,gens,loss,out_path):
	train=[data_i.split()[0] for data_i in datasets]
	names=list(train[0].keys())
	lines=[]
	for info_i,gen_i in gens.items():
		votes_i=get_votes(train,clf,names,gen_i)
		loss_i={name_j:loss_j(votes_i) for name_j,loss_j in loss.items()}
		for info_j,loss_j in loss_i.items():
			weights=optimize(loss_j,len(votes_i[0]))
			result_j=test_weights(datasets,clf,weights)
			metrics=exp.get_metrics(result_j)
			line_ij="%s,%s,%s" % (info_i,info_j,metrics)
			print(line_ij)
			lines.append(line_ij)
	if(out_path):
		files.save_txt(lines,out_path)
	return lines

def get_votes(train,clf,names,selector_gen):
	all_votes=[]
	for selector_i in selector_gen(names):
		results=learn.train_ens(train,clf=clf,selector=selector_i)
		all_votes.append(ens.Votes(results))
	return all_votes

def test_weights(datasets,clf,weights):
	votes=ens.Votes(learn.train_ens(datasets,clf))
	return votes.weighted(weights)

def optimize(loss_func,n_votes):
	starting_values = [0.5]*n_votes#len(all_votes[0])
	cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
	bounds = [(0,1)]*n_votes#len(all_votes[0])
	res = minimize(loss_func, starting_values, 
		method='SLSQP', bounds=bounds, constraints=cons)
	print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
	print('Best Weights: {weights}'.format(weights=res['x']))
	return res['x']

if __name__ == "__main__":
	dataset="3DHOI"
	dir_path="../../ICSS"#%s" % dataset
	paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
	paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
	aue_exp(paths["common"],paths["binary"],clf="LR",out_path="kfold.csv")