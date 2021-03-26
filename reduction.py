from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
import files,ens,feats

def make_selected_votes(common_path,binary_path,clf="LR",n_common=1400,n_binary=0):
	read=SepSelected(n_common,n_binary)
	return ens.make_votes(common_path,binary_path,clf,read)

def acc_curve(common_path,binary_path,clf="LR",n=15,step=100):
	acc=[]
	n_feats=step
	for i in range(1,n):
		print(i)
		votes=make_selected_votes(common_path,binary_path,clf,n_common=i*step)
		result_i=votes.voting()
		acc.append(result_i.get_acc())
		print(acc)
	return acc

class SelectedDataset(object):
	def __init__(self,n_feats):
		self.n_feats=n_feats

	def __call__(self,common_path,deep_path):
		datasets=ens.read_dataset(common_path,deep_path)
		for data_i in datasets:
			data_i.norm()
		return [reduce(data_i,n=self.n_feats)
		 				for data_i in datasets]

class SepSelected(object):
	def __init__(self,n_common=500,n_binary=84):
		self.n_common=n_common
		self.n_binary=n_binary

	def  __call__(self,common_path,deep_path):
		common=ens.read_dataset(common_path,None)[0]
		common.norm()
		common=reduce(common,n=self.n_common)
		if(not deep_path):
			return [common]
		binary=ens.read_dataset(None,deep_path)
		for data_i in binary:
			data_i.norm()
		binary=[reduce(data_i,n=self.n_binary)
		 				for data_i in binary]
		return [ common+binary_i for binary_i in binary]


def selected_deep(in_path,out_path):
    datasets=ens.read_dataset(None,in_path)
    files.make_dir(out_path)
    for i,data_i in enumerate(datasets):
    	data_i.norm()
    	new_data_i=reduce(data_i,n=84)
    	new_data_i.save("%s/%d" % (out_path,i))

def selected_common(paths,out_path,n=500):
	dataset=feats.read_unified(paths)
	dataset.norm()
	dataset=reduce(dataset,n)
	dataset.save(out_path)

def reduce(data_i,n=100):
	if( not n or n>data_i.dim()[0]):
		return  data_i
	print("Old dim:" + str(data_i.dim()))
	X,y,names=data_i.as_dataset()
	train_i=data_i.split()[0]
	new_X=recursive(train_i,data_i,n)
	new_data_i=feats.Feats()
	for j,name_j in enumerate(names):
		new_data_i[name_j]=new_X[j]
	print("New dim:" + str(new_data_i.dim()))
	return new_data_i

def lasso(train_i):
	X_train,y_train,names_train=train_i.as_dataset()
	clf = linear_model.Lasso(alpha=0.001,max_iter=1000)
	clf.fit(X_train,y_train)
	model = SelectFromModel(clf, prefit=True)
	new_X= model.transform(X)
	return new_X

def recursive(train_i,full_i,n=84):
	X_train,y_train,names_train=train_i.as_dataset()
	svc = SVC(kernel='linear',C=1)
	rfe = RFE(estimator=svc,n_features_to_select=n,step=10)
	rfe.fit(X_train,y_train)
	X,y,names=full_i.as_dataset()
	new_X= rfe.transform(X)
	return new_X

dataset="MSR"
dir_path="../dtw_paper/%s" % dataset
#binary="%s/binary/1D_CNN/dropout_0.5/feats" % dir_path
binary="../ActionClassifier/agum/agum2/feats"
common=files.top_files("%s/common/feats" % dir_path)
common=["%s/dtw" % common_i for common_i in common]
#acc=acc_curve(common,binary,clf="LR",n=35,step=50)
#print(acc)
selected_common(common,"MSR_100",n=100)