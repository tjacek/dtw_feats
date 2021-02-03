from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
#import exp,selection,feats,ens,learn
import files,ens,feats

#def multi_exp(dtw,deep,binary,out_path,clf="LR"):
#	fun=get_fun("feats")
#	fun(dtw,None,out_path,clf=clf)
#	fun(deep,None,out_path,clf=clf)
#	fun(None,binary,out_path,clf=clf)
#	fun(dtw,binary,out_path,clf=clf)
#	fun(deep+dtw,binary,out_path,clf=clf)

#def get_fun(gen_type):
#		def fun(dtw,binary,out_path,clf):
#			gen=selection.basic_selection
#			return exp.single_exp(dtw,binary,out_path,clf=clf,fun=gen)
#	elif(gen_type=="feats"):
#		def fun(dtw,binary,out_path,clf):
#			gen=selected_feats
#			return exp.single_exp(dtw,binary,out_path,clf=clf,fun=gen)
#	else:
#		def fun(dtw,binary,out_path,clf):
#			return exp.single_exp(dtw,binary,out_path,clf=clf)
#	return fun

#def selected_feats(common_path,binary_path,clf="SVC"):
#    datasets=ens.read_dataset(common_path,binary_path)
#    datasets=[reduce(data_i) for data_i in datasets]
#    results=[learn.train_model(data_i,clf_type=clf,binary=False)
#                for data_i in datasets]
#    return ens.Votes(results)

def selected_deep(in_path,out_path):
    datasets=ens.read_dataset(None,in_path)
    files.make_dir(out_path)
    for i,data_i in enumerate(datasets):
    	data_i.norm()
    	new_data_i=reduce(data_i)
    	new_data_i.save("%s/%d" % (out_path,i))

def reduce(data_i):
	print("Old dim:" + str(data_i.dim()))
	X,y,names=data_i.as_dataset()
	train_i=data_i.split()[0]
	new_X=recursive(train_i,data_i)
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

selected_deep("../dtw/deep","../dtw/RFE/deep")