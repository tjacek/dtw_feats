import sys
sys.path.append("..")
import files,ens,pairs

def basic_exp(in_path):
	paths=files.get_paths(in_path,name="dtw")
	result=ens.ensemble(paths[0],None,clf="LR",binary=False)[0]
	result.report()

def read_pairs(in_path):
	if(type(in_path)==list):
		return [ read_pairs(path_i) for path_i in in_path]
	raw_i=pairs.read(in_path)
	dict_i={ files.Name(name_i):data_i 
			for name_i,data_i in raw_i.items()}
	return pairs.DTWpairs(dict_i)

def get_preferences(pair_i):
	train,test=pair_i.split()
	ord_i=pair_i.ordering(test[0],train)
	cats_i=[name_i.get_cat() for name_i in ord_i ]
	print(cats_i)

in_path="../../ICSS/MHAD/dtw/"#corl/pairs"
paths=files.get_paths(in_path,name="pairs")
dtw_pairs=read_pairs(paths)
get_preferences(dtw_pairs[0])