import sys
sys.path.append("..")
import numpy as np,random
import ens,feats,learn,files,exp

def random_selection(common_path,binary_path,n,n_clf,clf="LR",fun=None):
	if(not fun):
		fun=ens.make_votes
	votes=fun(common_path,binary_path,clf)
	if(n_clf>len(votes)):
		n_clf=len(votes)
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

def selection_exp(common,binary,cf_path=None):
	s_clf=random_selection(common,binary,1000,27,clf="LR")
	print(s_clf)
	result,votes=ens.ensemble(common,binary,
		clf="LR",binary=False,s_clf=s_clf)
	result.report()
	if(cf_path):
		result.get_cf(cf_path)

if __name__ == "__main__":
	dataset="MSR"
	dir_path="../../dtw_paper/%s" % dataset
	binary="%s/sim/feats" % dir_path
	common="%s/common/%s_500" % (dir_path,dataset)
	selection_exp(common,binary,"MHAD")