import sys
sys.path.append("..")
import numpy as np
from sklearn.metrics import accuracy_score
import files,ens,pairs

def basic_exp(in_path):
	paths=files.get_paths(in_path,name="dtw")
	result=ens.ensemble(paths[0],None,clf="LR",binary=False)[0]
	result.report()

def voting(in_path):
	paths=files.get_paths(in_path,name="pairs")
	dtw_pairs=read_pairs(paths)
	pref_dicts=[ get_preferences(pairs_i) 
			for pairs_i in dtw_pairs]
	names=pref_dicts[0].keys()

	y_pred,y_true=[],[]
	for name_i in names:
		y_true.append(name_i.get_cat())
		votes=[ pref_j[name_i] for pref_j in pref_dicts]
		y_pred.append(border_count(votes))
	print(accuracy_score(y_true,y_pred))

def border_count(pref):
	n_cats=len(pref[0])
	votes=np.zeros((n_cats,))
	for pref_i in pref:
		for i,cat_i in enumerate(pref_i):
			votes[cat_i]+=(n_cats-i)
	return np.argmax(votes)

def read_pairs(in_path):
	if(type(in_path)==list):
		return [ read_pairs(path_i) for path_i in in_path]
	raw_i=pairs.read(in_path)
	dict_i={ files.Name(name_i):data_i 
			for name_i,data_i in raw_i.items()}
	return pairs.DTWpairs(dict_i)

def get_preferences(pair_i):
	train,test=pair_i.split()
	pref_dict={}
	for name_i in test:
		ord_i=pair_i.ordering(name_i,train)
		cats=[name_j.get_cat() for name_j in ord_i ]
		pref_dict[name_i]=unique_elem(cats)
	return pref_dict

def unique_elem(cats):
	unique=[]
	for cat_j in cats:
		if(not unique.count(cat_j)):
			unique.append(cat_j)
	return unique

in_path="../../ICSS/MHAD/dtw/"#corl/pairs"
voting(in_path)