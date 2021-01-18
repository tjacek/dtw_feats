import numpy as np
from collections import Counter

class KNN(object):
	def __init__(self,names,pairs):
		self.pairs=pairs
		self.names=names

	def __call__(self,name_i,k=3):
		neighbour=self.neighbors(name_i,k)
		cats=[exmple_j.get_cat() 
				for exmple_j in neighbour]
		cat_pred=Counter(cats).most_common()[0][0]
		return cat_pred

	def neighbors(self,name_i,k=3):
		distance=self.pairs.features(name_i,self.names)
		indexes=np.argsort(distance)[:k]
		return [self.names[i] for i in indexes]

def knn_selection(names,pairs,k=3):		
	knn=KNN(names,pairs)
	s_names=[]
	for name_i in names:
		if(name_i.get_cat()==knn(name_i)):
			s_names.append(name_i)
	return s_names