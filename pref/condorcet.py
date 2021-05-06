import sys
sys.path.append("..")
import numpy as np
from ens import Votes
from sklearn.metrics import accuracy_score
import pref

def meta_ensemble(main_path,minor_path):
	y_true,main_pred,main_condor=get_condor(main_path)
	y_true,minor_pred,minor_condor=get_condor(minor_path)
	y_meta=[]
	for i,main_i in enumerate(main_pred):
		if((not main_condor[i]) and (minor_condor[i])):
			print(y_true[i])
			print(minor_pred[i])
			print("&&&&")
			y_meta.append(minor_pred[i])
		else:
			y_meta.append(main_i)
	print(accuracy_score(y_true,y_meta))

def count_condor(in_path):
	y_true,y_pred,condor=get_condor(in_path)
	print(accuracy_score(y_true,y_pred))
	print(np.mean(condor))
	print(len(condor)-np.sum(condor))
	error=error_cond(y_true,y_pred,condor)
	print(np.mean(error))
	print(len(error)-np.sum(error))

def get_condor(in_path):
	votes=pref.read_pref(in_path)
	result=votes.voting(False)
	y_true,y_pred=result.as_labels()
	votes=pref.prepare_votes(votes)
	votes=[ pref.to_preference(vote_i) 
			for vote_i in votes]
	condor=np.array([is_condorcet_winner(win_i,votes[i]) 
						for i,win_i in enumerate(y_pred)])
	condor=condor.astype(int)
	return y_true,y_pred,condor

def is_condorcet_winner(win_i,pref_i):
	n_cats=pref_i.order.shape[0]
	for cat_i in range(n_cats):
		if(cat_i!=win_i):
			if(not pref_i.pairwise_score(win_i,cat_i)):
				return False
	return True

def error_cond(y_true,y_pred,condor):
	error=[]
	for i,cond_i in enumerate(condor):
		if(y_true[i]!=y_pred[i]):
			error.append(cond_i)
	return error

dataset="3DHOI"
#count_condor("../s_LR/%s" % dataset)
meta_ensemble("../s_LR/%s" % dataset,"../s_SVC/%s" % dataset)