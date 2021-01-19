import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import ens

def acc_hist(common_path,binary_path,clf="LR"):
	votes=ens.make_votes(common_path,binary_path,clf=clf)
	acc=votes.indv_acc()
	show_histogram(acc,title='indiv acc',cumsum=False)

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
acc_hist(None,binary,clf="LR")