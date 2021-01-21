import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
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

dataset="MHAD"
path='../ICSS_exp/%s/' % dataset
binary='%s/ens/lstm_gen/feats' % path
acc_hist(None,binary,clf="LR",cat_i=5)