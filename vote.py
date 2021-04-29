import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter
import ens,files

class Preferences(object):
	def __init__(self, order):
		self.order=np.array(order)

	def n_votes(self):
		return self.order.shape[0]

	def by_vote(self):
		return [vote_i for vote_i in self.order]

	def by_order(self):
		return [ord_i for ord_i in self.order.T]

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
		ord_i= np.argsort(vote_j)
		pref.append(ord_i)
	return Preferences(pref)

def borda_count(prefer):
	raise Exception( prefer.by_order())
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

def k_aproval(prefer,k=1):
	prefer=np.array(prefer)
	aprov=prefer[:,-k:].flatten()
	votes=Counter(aprov)
	return votes.most_common()[0][0]

def coombs(prefer):
	n_cats=prefer[0].shape[0]
	major=n_cats/2
	votes=[Counter(prefer_i) 
		for prefer_i in np.array(prefer).T]

	votes.reverse()
	vote_i=votes[0]
	cat_i=vote_i.most_common()[0]
	count_i=cat_i[1]
	if(count_i>major):
		print(cat_i)
		return cat_i[0]
	raise Exception(count_i)

dataset="3DHOI"
dir_path="../ICSS/%s" % dataset
common="%s/dtw" % dir_path
common=files.get_paths(common,name="dtw")
common.append("%s/1D_CNN/feats" % dir_path)
binary="%s/ens/feats" % dir_path 
ensemble(common,binary,system=None,clf="LR")