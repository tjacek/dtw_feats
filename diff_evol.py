import numpy as np
from scipy.optimize import differential_evolution
import ens,learn,exp,files

class LossFunction(object):
	def __init__(self, votes,squared=False):
		self.votes=votes
		self.squared=squared
	
	def __call__(self,weights):	
		norm=weights/np.sum(weights)
		result=self.votes.weighted(norm)
		if(self.squared):
			return (1.0-result.get_acc())
		return 1.0-result.get_acc()

def diff_exp(dtw,deep,binary1,binary2):
	paths=[ (deep,binary1),(deep+dtw,binary1), 
			(deep,[binary1,binary2]),(deep+dtw,[binary1,binary2])]
	results=[diff_voting(common_i,binary_i,clf="LR")
				 for common_i,binary_i in paths]
	for path_i,result_i in zip(paths,results):
		info_i=exp.exp_info(path_i[0],path_i[1],result_i)
		print("%s,%s,%s" % info_i)

def diff_voting(common,deep,clf="LR"):
	datasets=ens.read_dataset(common,deep)
	weights=find_weights(datasets)
	results=learn.train_ens(datasets,clf="LR")
	votes=ens.Votes(results)
	result=votes.weighted(weights)
	return result

def find_weights(datasets):
	results=validation_votes(datasets)
	loss_fun=LossFunction(ens.Votes(results))
	bound_w = [(0.0, 1.0)  for _ in datasets]
	result = differential_evolution(loss_fun, bound_w, maxiter=1000, tol=1e-7)
	weights=result['x']
	return weights

def validation_votes(datasets,clf="LR"):
	results=[]
	for data_i in datasets:
		data_i.norm()
		train=data_i.split()[0]
		clf_i=learn.make_model(train,clf)
		y_pred=clf_i.predict_proba(train.get_X())
		result_i =learn.Result(train.get_labels(),y_pred,train.names())
		results.append(result_i)
	return results

if __name__ == "__main__":
	dataset="3DHOI"
#	dir_path="../ICSS_exp" 
#	paths=exp.basic_paths(dataset,dir_path,"dtw","ens/lstm/feats")
#	paths["common"].append("%s/common/1D_CNN/feats" % paths["dir_path"])
#	paths["common"]="s_feats2/%s_200" % dataset
	binary="../ICSS_exp/%s/ens/lstm/feats"
	paths=exp.common_paths("s_feats2",binary)
	diff_exp=exp.BasicExp(diff_voting)
	lines=diff_exp(paths,"diff_l1")
	print(lines)
#	result=diff_voting(paths["common"],paths["binary"],clf="LR")
#	result.report()