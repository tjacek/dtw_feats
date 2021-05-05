import sys
sys.path.append("..")
import numpy as np
from ens import Votes
from sklearn.metrics import accuracy_score
import pref

def count_condor(in_path):
	votes=pref.read_pref(in_path)
	result=votes.voting(False)
	y_true,y_pred=result.as_labels()
	votes=pref.prepare_votes(votes)
	votes=[ pref.to_preference(vote_i) 
			for vote_i in votes]
	condor=[1-int(is_condorcet_winner(win_i,votes[i])) 
				for i,win_i in enumerate(y_pred)]
	print(accuracy_score(y_true,y_pred))
	print(np.mean(condor))
	print(np.sum(condor))
	error=error_cond(y_true,y_pred,condor)
	print(np.mean(error))
	print(np.sum(error))

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



dataset="MSR"
count_condor("../s_SVC/%s" % dataset)