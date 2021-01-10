from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
import exp,selection,feats,ens,learn

def multi_exp(dtw,deep,binary,out_path,clf="LR"):
	fun=get_fun("feats")
	fun(dtw,None,out_path,clf=clf)
	fun(deep,None,out_path,clf=clf)
	fun(None,binary,out_path,clf=clf)
	fun(dtw,binary,out_path,clf=clf)
	fun(deep+dtw,binary,out_path,clf=clf)

def get_fun(gen_type):
	if(gen_type=="selection"):
		def fun(dtw,binary,out_path,clf):
			gen=selection.basic_selection
			return exp.single_exp(dtw,binary,out_path,clf=clf,fun=gen)
	elif(gen_type=="feats"):
		def fun(dtw,binary,out_path,clf):
			gen=selected_feats
			return exp.single_exp(dtw,binary,out_path,clf=clf,fun=gen)
	else:
		def fun(dtw,binary,out_path,clf):
			return exp.single_exp(dtw,binary,out_path,clf=clf)
	return fun

def selected_feats(common_path,binary_path,clf="SVC"):
    datasets=ens.read_dataset(common_path,binary_path)
    datasets=[reduce(data_i) for data_i in datasets]
    results=[learn.train_model(data_i,clf_type=clf,binary=False)
                for data_i in datasets]
    return ens.Votes(results)

def reduce(data_i):
	print("Old dim:" + str(data_i.dim()))
	X,y,names=data_i.as_dataset()
	train_i=data_i.split()[0]
	X_train,y_train,names_train=train_i.as_dataset()
	clf = linear_model.Lasso(alpha=0.001,max_iter=1000)
	clf.fit(X_train,y_train)
	model = SelectFromModel(clf, prefit=True)
	new_X= model.transform(X)
	new_data_i=feats.Feats()
	for j,name_j in enumerate(names):
		new_data_i[name_j]=new_X[j]
	print("New dim:" + str(new_data_i.dim()))
	return new_data_i

deep=['../ICSS_exp/3DHOI/common/1D_CNN/feats']
binary='../ICSS_exp/3DHOI/ens/lstm/feats'
dtw=['../ICSS_exp/3DHOI/dtw/corl/person', '../ICSS_exp/3DHOI/dtw/max_z/person']

out_path="reduction/3DHOI_lasso"
multi_exp(dtw,deep,binary,out_path)
exp.show_result(out_path,hard=False)
