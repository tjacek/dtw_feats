import sys
sys.path.append("..")
import files,ens,pairs

def basic_exp(in_path):
	paths=files.get_paths(in_path,name="dtw")
	result=ens.ensemble(paths[0],None,clf="LR",binary=False)[0]
	result.report()

def get_preferences(pair_i):
	train,test=pair_i.split()
	ord_i=pair_i.ordering(test[0],train)
	print(ord_i)


in_path="../../ICSS/MHAD/dtw/"#corl/pairs"
paths=files.get_paths(in_path,name="pairs")
dtw_pairs=pairs.read(paths)
get_preferences(dtw_pairs[0])