import numpy as np
from collections import defaultdict 

def select_names(names,pairs,fun):
	group=group_name(names,fun)
	s_names=[]
	for name_i in group.keys():
		dist_i=pairs.dist_matrix(group[name_i])
		k=np.argmax(np.sum(dist_i,axis=0))
		s_names.append(group[name_i][k])
	return s_names

def group_name(names,fun):
	group = defaultdict(lambda:[])
	for name_i in names:
		group[fun(name_i)].append(name_i)
	return group

def get_id(name_i):
	return '_'.join(name_i.split("_")[:1])