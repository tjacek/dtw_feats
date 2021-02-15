import numpy as np
import seaborn
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import ens

def acc_hist(common_path,binary_path,clf="LR",cat_i=None):
	votes=ens.make_votes(common_path,binary_path,clf=clf)
	if(cat_i is None):
		acc=votes.indv_acc()
		title='indiv acc'
	else:
		acc_matrix=votes.acc_matrix()
		acc=acc_matrix[cat_i]
		title='acc cat %d' % cat_i
	show_histogram(acc,title=title,cumsum=False)

def show_histogram(hist,title='hist',cumsum=True):
	if(type(hist)==list):
		hist=np.array(hist)
	if(cumsum):
		hist=np.cumsum(hist)
	fig = plt.figure()
	x=range(hist.shape[0])
	plt.bar(x,hist)
	fig.suptitle(title)
	plt.show()

def acc_matrix(common_path,binary_path,clf="LR"):
	votes=ens.make_votes(common,binary,clf="LR")
	acc=votes.acc_matrix()
	ax=seaborn.heatmap(acc)
	plt.show()
	plt.clf()

def hand_selection(ordering,common_path,binary_path,clf="LR"):
	votes=ens.make_votes(common_path,binary_path,clf="LR")
	s_votes=ens.Votes([votes.results[i] for i in ordering])
	s_result=s_votes.voting()
	s_result.report()

common="s_dtw"
binary="../clean3/agum/ens/basic/feats"
#acc_hist(common,binary,clf="LR",cat_i=6)
order=[0, 1, 3, 4, 6, 8, 9, 10]
hand_selection(order,common,binary)
acc_matrix(common,binary,clf="LR")
