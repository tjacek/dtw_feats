import numpy as np
from scipy.optimize import differential_evolution
import ens

class LossFunction(object):
	def __init__(self, votes):
		self.votes=votes
	
	def __call__(self,weights):	
		norm=weights/np.sum(weights)
		result=self.votes.weighted(norm)
		return 1.0-result.get_acc()

def diff_voting(common,deep,clf="LR"):
	datasets=ens.read_dataset(None,deep)
	votes=ens.make_votes(common,binary,clf=clf)	
	loss_fun=LossFunction(votes)
	bound_w = [(0.0, 1.0)  for _ in datasets]
	result = differential_evolution(loss_fun, bound_w, maxiter=1000, tol=1e-7)
	weights=result['x']
	result=votes.weighted(weights)
	print(result.names)
	print(result.get_acc())

if __name__ == "__main__":
    dataset="MHAD"
    path='../ICSS_exp/%s/' % dataset
    deep=['%s/common/1D_CNN/feats' % path]
    binary='%s/ens/lstm_gen/feats' % path
#    binary='../ICSS_sim/%s/sim/feats' % dataset
#    binary=['%s/ens/lstm_gen/feats' % path,'../ICSS_sim/%s/sim/feats' % dataset]
    dtw=['%s/dtw/corl/dtw' % path, '%s/dtw/max_z/dtw' % path]
    diff_voting(deep,binary,clf="LR")

