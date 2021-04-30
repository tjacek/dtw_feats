import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from collections import Counter
import ens,files

class Preferences(object):
	def __init__(self, order):
		self.order=np.array(order)

	def n_votes(self):
		return self.order.shape[0]

	def empty_votes(self):
		return np.zeros((self.n_votes(),))

	def by_vote(self):
		return [vote_i for vote_i in self.order]

	def by_order(self,as_counter=False,flip=False):
		if(as_counter):
			ordering=[Counter(ord_i) for ord_i in self.order.T]
		else:
			ordering=[ord_i for ord_i in self.order.T]
		if(flip):
			ordering.reverse()
		return ordering

def ensemble(common_path,binary_path,system=None,clf="LR",cf_path=None):
	if(system is None):
		system=borda_count
	votes=ens.make_votes(common_path,binary_path,clf,None)
	y_true=votes.results[0].y_true
	votes=prepare_votes(votes)
	y_pred=[]
	for vote_i in votes:
		pref_i= to_preference(vote_i)
		y_pred.append(system(pref_i))
	print(classification_report(y_true,y_pred,digits=4))
	print(accuracy_score(y_true,y_pred))
	if(cf_path):
		cf_matrix=confusion_matrix(y_true,y_pred)
		np.savetxt(cf_path,cf_matrix,delimiter=",",fmt='%.2e')

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
	votes=prefer.empty_votes()
	for prefer_i in prefer.by_vote():
		for j,cat_j in enumerate(prefer_i):
			votes[cat_j]+=j
	return np.argmax(votes)

def bucklin(prefer):
	votes=prefer.empty_votes()
	half=votes.shape[0]/2
	for count_i in prefer.by_order(True,True):
		for cat,n in count_i.items():
			votes[cat]+=n
			if(np.amax(votes)>=half):
				return np.argmax(votes)

def k_aproval(prefer,k=2):
	aprov=prefer.order[:,-k:].flatten()
	votes=Counter(aprov)
	return votes.most_common()[0][0]

def coombs(prefer):
	counters=prefer.by_order(as_counter=True,flip=True)
	first=counters[0]
	last=counters.pop()
	while(first!=last):
		if(sum(last.values())==0):
			last=counters.pop()
		best=first.most_common()[0]
		major=sum(first.values())/2
		if(major<best[1]):
			print(best)
			return best[0]
		worst= last.most_common()[0]
		first[worst[0]]=0
		last[worst[0]]=0
		print(last)
	raise Exception(first)

dataset="3DHOI"
dir_path="../ICSS/%s" % dataset
common="%s/dtw" % dir_path
common=files.get_paths(common,name="dtw")
common.append("%s/1D_CNN/feats" % dir_path)
binary="%s/ens/feats" % dir_path 
ensemble(common,binary,system=coombs,clf="LR",cf_path="3DHOI")