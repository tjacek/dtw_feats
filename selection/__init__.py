import sys
sys.path.append("..")
import numpy as np,random
import ens,feats,learn,files,exp

def random_selection(common_path,binary_path,n,n_clf,clf="LR",fun=None):
	if(not fun):
		fun=ens.make_votes
	votes=fun(common_path,binary_path,clf)
	indexes=[i for i in range(n_clf)]
	def helper(size):
		ind_i= random.sample(indexes,size)
		subset_i=[votes.results[k] for k in ind_i]
		votes_i=ens.Votes(subset_i)
		result_i=votes_i.voting(False)
		return result_i.get_acc(),ind_i
	best_set=[]
	best_acc=[]
	for k in range(2,n_clf):
		acc,ind=list(zip(*[helper(k) for i in range(n)]))
		t=np.argmax(acc)
		best_acc.append(acc[t])
		print(acc[t])
		ind[t].sort()
		print(ind[t])
		best_set.append(ind[t])
	return best_set[np.argmax(best_acc)]

if __name__ == "__main__":
	dataset="../../dtw_paper/set2"
	common="%s/common/s_dtw" % dataset
	binary="%s/binary/1D_CNN/feats" %dataset
	random_selection(common,binary,1000,12)