import numpy as np
from scipy.optimize import differential_evolution
import ens,learn

class LossFunction(object):
	def __init__(self, votes):
		self.votes=votes
	
	def __call__(self,weights):	
		norm=weights/np.sum(weights)
		result=self.votes.weighted(norm)
		return 1.0-result.get_acc()

def diff_voting(common,deep,clf="LR"):
	datasets=ens.read_dataset(common,deep)
	weights=find_weights(datasets)
	results=learn.train_ens(datasets,clf="LR")
	votes=ens.Votes(results)
	result=votes.weighted(weights)
	print(result.get_acc())
	
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
		results.append(result_i)#print(result_i.get_acc())
	return results

if __name__ == "__main__":
    dataset="MHAD"
    path='../ICSS_exp/%s/' % dataset
    deep=['%s/common/1D_CNN/feats' % path]
    binary='%s/ens/lstm_gen/feats' % path
#    binary='../ICSS_sim/%s/sim/feats' % dataset
#    binary=['%s/ens/lstm_gen/feats' % path,'../ICSS_sim/%s/sim/feats' % dataset]
    dtw=['%s/dtw/corl/dtw' % path, '%s/dtw/max_z/dtw' % path]
    diff_voting(deep,binary,clf="LR")

