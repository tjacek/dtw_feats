import sys
sys.path.append("..")
import numpy as np
from collections import Counter
import pickle
import ens,files,learn
from ens import Votes
from pref.systems import *

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

	def pairwise_score(self,x,y):
		score=0
		for vote_i in self.by_vote():
			x_index=np.where(vote_i==x)[0][0]
			y_index=np.where(vote_i==y)[0][0]
			if(y_index<x_index):
				score+=1
		return score > (self.n_votes()/2)

def read_pref(in_path):
	with open(in_path, 'rb') as handle:
		return pickle.load(handle)

def ensemble(common_path,binary_path,system=None,clf="LR",s_clf=None):
    votes=ens.make_votes(common_path,binary_path,clf,None)
    if(s_clf):
        votes=ens.Votes([votes.results[clf_i] 
          	for clf_i in s_clf])
    return voting(votes,system)

def ext_exp(in_path,system,cf_path=None):
	return voting(read_pref(in_path),system,cf_path)

def voting(votes,system,cf_path=None):
	if(system is None):
		system=borda_count
	y_true=votes.results[0].y_true
	names=votes.results[0].names
	votes=prepare_votes(votes)
	y_pred=[]
	for vote_i in votes:
		pref_i= to_preference(vote_i)
		y_pred.append(system(pref_i))
	return learn.Result(y_true,y_pred,names)

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

if __name__ == "__main__":
    dataset="3DHOI"
    dir_path="../../ICSS/%s" % dataset
    common="%s/dtw" % dir_path
    common=files.get_paths(common,name="dtw")
    common.append("%s/1D_CNN/feats" % dir_path)
    binary="%s/ens/feats" % dir_path 
    result=ensemble(common,binary,system=coombs,clf="LR",s_clf=[1,6,7])
    result.report()
#	raise Exception("End")
#	result=ext_exp("../3DHOI",None)
#	result.report()