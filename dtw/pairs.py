import numpy as np
import pickle
from dtaidistance import dtw, dtw_ndim
import files

class DTWpairs(object):
	def __init__(self,pairs):
		self.pairs=pairs

	def keys(self):
		return self.pairs.keys()

	def features(self,name_i,names):
		return np.array([self.pairs[name_i][name_j]  
					for name_j in names])
 	
	def set(self,key1,key2,data_i):
 		self.pairs[key1][key2]=data_i
	
	def split(self,selector=None):
		if(selector is None):
			selector=files.person_selector
		train,test=[],[]
		for name_i in self.pairs.keys():
			if(selector(name_i)):
				train.append(name_i)
			else:
				test.append(name_i)
		return train,test

	def dist_matrix(self,names):
		distance=[[ self[name_i][name_j] for name_i]
					for name_j]
		return np.array(distance)

	def save(self,out_path):
		with open(out_path, 'wb') as handle:
			pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read(in_path):
	with open(in_path, 'rb') as handle:
		return pickle.load(handle)

def make_dtw_pairs(seqs):
	pairs={ name_i:{name_i:0.0}
				for name_i in seqs.keys()}
	return DTWpairs(pairs)

class Seqs(dict):
	def __init__(self, arg=[]):
		super(Seqs, self).__init__(arg)

	def split(self):
		train,test=files.split(self)
		return Seqs(train),Seqs(test)

def compute_pairs(in_path,out_path):
	seqs=read_seqs(in_path)
	pairs=make_pairwise_distance(seqs)
	pairs.save(out_path)

def read_seqs(in_path):
	seqs=Seqs()
	for path_i in files.top_files(in_path):
		data_i=np.loadtxt(path_i, delimiter=',')
		name_i=path_i.split('/')[-1]
		name_i=files.Name(name_i).clean()#clean(name_i)
		seqs[name_i]=data_i
	return seqs

def make_pairwise_distance(seqs):
	dtw_pairs=make_dtw_pairs(seqs)
	names=list(seqs.keys())
	n_ts=len(names)
	for i in range(1,n_ts):
		print(i)
		for j in range(0,i):
			name_i,name_j=names[i],names[j]
			distance_ij=dtw_ndim.distance(seqs[name_i],seqs[name_j])
			dtw_pairs.set(name_i,name_j,distance_ij)
			dtw_pairs.set(name_j,name_i,distance_ij)
	return dtw_pairs