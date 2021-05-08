import sys
sys.path.append("..")
from sklearn.cross_validation import StratifiedShuffleSplit
import ens,learn

def diff_voting(common,deep,clf="LR"):
	datasets=ens.read_dataset(common,deep)
	results=learn.train_ens(datasets,clf="LR",selector=strat_selector)


def strat_selector(names):
	raise Exception(type(names))

dataset="3DHOI"
dir_path="../ICSS"#%s" % dataset
paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
diff_voting(paths["common"],paths["deep"],clf="LR")