import numpy as np
import seaborn
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import files,ens,exp,learn

def acc_hist(common_path,binary_path,clf="LR",
			out_path=None,title='3DHOI'):
	votes=full_train(common_path,binary_path,clf)
	files.make_dir(out_path)
	fun=simple_histogram
	for i,result_i in enumerate(votes.results):
		hist_i=result_i.indv_acc()
		title_i="%s_%d" % (title,i)
		plot_i=fun(hist_i,title=title_i,cumsum=False,show=False)
		plot_i.savefig("%s/%d" % (out_path,i))

def multi_acc_hist(common_path,binary_path,clf="LR",
			out_path=None,title='Accuracy'):
	base_votes=base_train(common_path,binary_path,clf)
	full_votes=full_train(common_path,binary_path,clf)
	for i,result_i in enumerate(base_votes.results):
		base_hist_i=result_i.indv_acc()
		full_hist_i=full_votes.results[i].indv_acc()
		plot_i=multi_histogram([full_hist_i,base_hist_i],
			colors=['b','r'],legend=['val_acc','test_acc'])
		plot_i.savefig("%s/%d" % (out_path,i))

def base_train(common,binary,clf):
	datasets=ens.read_dataset(common,binary)
	results=learn.train_ens(datasets,clf=clf)
	return ens.Votes(results)

def full_train(common,binary,clf):
	datasets=ens.read_dataset(common,binary)
	results=[]
	for data_i in datasets:
		train,test=data_i.split()
		model=learn.make_model(train,clf)
		X_train,y_true=train.get_X(),train.get_labels()
		y_pred=model.predict_proba(X_train)
		result_i=learn.Result(y_true,y_pred,test.names())
		results.append(result_i)
	return ens.Votes(results)

def simple_histogram(hist,title='hist',cumsum=True,show=True):
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

def multi_histogram(hists,colors,title="Accuracy",legend=None):
	ax = plt.figure()#.subplot(111)
	for i,hist_i in enumerate(hists):
		hist_i=np.array(hist_i)
		x=np.array(range(hist_i.shape[0]))
		plt.bar(x+i*0.2, hist_i,  width=0.2,color=colors[i])
		print(legend[i])
		plt.legend(legend)#[i])
	ax.suptitle(title)	
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
multi_acc_hist(paths["common"],paths["binary"],out_path="hist")