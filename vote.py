import numpy as np
import ens

def ensemble(common_path,binary_path,system,clf="LR"):
	votes=ens.make_votes(common_path,binary_path,clf,None)
	votes=prepare_votes(votes)
	print(len(votes))

def prepare_votes(votes):
	y_true=votes.results[0].y_true
	n_samples=len(y_true)
	votes=np.array([ result_i.as_numpy() 
		for result_i in votes.results])
	return [votes[:,i,:] for i in range(n_samples) ]
    
dataset="MHAD"
dir_path="../ICSS/%s" % dataset
binary="%s/ens/feats" % dir_path
ensemble(None,binary,None,clf="LR")