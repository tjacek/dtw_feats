import numpy as np
import seaborn
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import ens,exp,learn

def acc_hist(common_path,binary_path,clf="LR",cat_i=None,cross=False):
	datasets=ens.read_dataset(common_path,binary_path)
	results=learn.train_ens(datasets,clf="LR")
	votes=ens.Votes(results)
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

dataset="3DHOI"
dir_path="../ICSS"#%s" % dataset
paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
acc_hist(paths["common"],paths["binary"])