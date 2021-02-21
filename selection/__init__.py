import sys
sys.path.append("..")
import numpy as np,random
import ens,feats,learn,files,exp

def random_selection(common_path,binary_path,n,n_clf,clf="LR"):
	votes=ens.make_votes(common_path,binary_path,clf,read=None)
	indexes=[i for i in range(n_clf)]
	def helper(size):
		ind_i= random.sample(indexes,size)
		subset_i=[votes.results[k] for k in ind_i]
		votes_i=ens.Votes(subset_i)
		result_i=votes_i.voting(False)
		return result_i.get_acc(),ind_i
	for k in range(2,n_clf):
		acc,ind=list(zip(*[helper(k) for i in range(n)]))
		t=np.argmax(acc)
		print(acc[t])
		ind[t].sort()
		print(ind[t])

if __name__ == "__main__":
	dataset="../../dtw_paper/set2"
	common="%s/common/s_dtw" % dataset
	binary="%s/binary/1D_CNN/feats" %dataset
	random_selection(common,binary,1000,12)