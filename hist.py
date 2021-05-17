import numpy as np
import seaborn
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import files,ens,exp,learn

def acc_hist(common_path,binary_path,clf="LR",
			out_path=None,title='3DHOI'):
	datasets=ens.read_dataset(common_path,binary_path)
	results=learn.train_ens(datasets,clf="LR")
	votes=ens.Votes(results)
	files.make_dir(out_path)
	for i,result_i in enumerate(votes.results):
		hist_i=result_i.indv_acc()
		title_i="%s_%d" % (title,i)
		plot_i=show_histogram(hist_i,title=title_i,cumsum=False,show=False)
		plot_i.savefig("%s/%d" % (out_path,i))

def show_histogram(hist,title='hist',cumsum=True,show=True):
	if(type(hist)==list):
		hist=np.array(hist)
	if(cumsum):
		hist=np.cumsum(hist)
	fig = plt.figure()
	x=range(hist.shape[0])
	plt.bar(x,hist)
	fig.suptitle(title)
	if(show):
		plt.show()
	return plt

def acc_matrix(common_path,binary_path,clf="LR"):
	votes=ens.make_votes(common,binary,clf="LR")
	acc=votes.acc_matrix()
	ax=seaborn.heatmap(acc)
	plt.show()
	plt.clf()

dataset="3DHOI"
dir_path="../ICSS"#%s" % dataset
paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
acc_hist(paths["common"],paths["binary"],out_path="hist")