import sys
sys.path.append("..")
import numpy as np
import ens

def success_selection(common_path,binary_path,clf="LR"):
    votes=ens.make_votes(common_path,binary_path,clf="LR")
    n_cats=len(votes)
    acc_matrix=[[ result_i.cat_acc(cat_i) 
    				for result_i in votes.results]
    					for cat_i in range(n_cats)]
    acc_matrix=np.array(acc_matrix)
    best=np.argmax(acc_matrix,axis=0)
    clfs=np.argsort(best)
    return clfs

def acc_curve(common,binary,clf,clfs):
	clf_selction=range(2,len(clfs))
	results=[ens.ensemble(common,binary,False,clf,clfs[:k])[0]
	 			for k in clf_selction]
	acc=[result_i.get_acc() for result_i in results]
	print(acc)

if __name__ == "__main__":
    dataset="../../dtw_paper/MSR"
    common="%s/common/MSR_500" % dataset
    binary="%s/binary/1D_CNN/feats" %dataset
    clfs=success_selection(common,binary,clf="LR")
    acc_curve(common,binary,"LR",clfs)