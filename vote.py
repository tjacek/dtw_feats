import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter
import ens,files

def ensemble(common_path,binary_path,system=None,clf="LR"):
	if(system is None):
		system=borda_count
	votes=ens.make_votes(common_path,binary_path,clf,None)
	y_true=votes.results[0].y_true
	votes=prepare_votes(votes)
	y_pred=[]
	for vote_i in votes:
		pref_i= to_preference(vote_i)
		y_pred.append(system(pref_i))
	print(accuracy_score(y_true,y_pred))

def prepare_votes(votes):
	y_true=votes.results[0].y_true
	n_samples=len(y_true)
	votes=np.array([ result_i.as_numpy() 
		for result_i in votes.results])
	return [votes[:,i,:] for i in range(n_samples) ]

def to_preference(vote_i):
	pref=[]
	for vote_j in vote_i: 
		ord_i= np.argsort(vote_j)#np.flip(np.argsort(vote_j))
		pref.append(ord_i)
	return pref

def borda_count(prefer):
	votes=np.zeros( (len(prefer),))
	for prefer_i in prefer:
		for j,cat_j in enumerate(prefer_i):
			votes[cat_j]+=j
	return np.argmax(votes)

def bucklin(prefer):
	prefer=np.flip(np.array(prefer).T)
	votes=np.zeros((prefer.shape[0],))
	half=votes.shape[0]/2
	for prefer_i in prefer:
		count_i=Counter(prefer_i)
		for cat,n in count_i.items():
			votes[cat]+=n
			if(np.amax(votes)>=half):
				return np.argmax(votes)
	raise Exception("OK")

dataset="MHAD"
dir_path="../ICSS/%s" % dataset
common="%s/dtw" % dir_path
common=files.get_paths(common,name="dtw")
common.append("%s/1D_CNN/feats" % dir_path)
binary="%s/ens/feats" % dir_path 
ensemble(common,binary,system=bucklin,clf="LR")