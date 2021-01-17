import numpy as np
from collections import Counter

def knn_selection(names,pairs,k=3):
	def knn_helper(name_i):
		distance=pairs.features(name_i,names)
		indexes=np.argsort(distance)[:k]
		return [names[i] for i in indexes]
	s_names=[]
	for name_i in names:
		neighbour=knn_helper(name_i)
		cats=[exmple_j.get_cat() 
				for exmple_j in neighbour]
		cat_pred=Counter(cats).most_common()[0][0]
		if(name_i.get_cat()==cat_pred):
			s_names.append(name_i)
	return s_names