import dtw.pairs,files,feats,dtw.knn
import files,learn

def compute_feats(pair_path,feat_path):
	pairs=dtw.pairs.read(pair_path)
	train=pairs.split()[0]
	dtw_feats=feats.Feats()
	for name_i in train:
		dtw_feats[name_i]=pairs.features(name_i,train)
	dtw_feats.save(feat_path)

def select_feats(pair_path,feat_path):
	pairs=dtw.pairs.read(pair_path)
	train=pairs.split()[0]
	s_names=dtw.knn.knn_selection(train,pairs,k=3)
	print(s_names)

def knn_clf(pair_path,k=3):
	pairs=dtw.pairs.read(pair_path)
	train,test=pairs.split()
	clf=dtw.knn.KNN(train,pairs)
	y_pred=[clf(name_i,k) for name_i in test]
	y_true=[name_i.get_cat() for name_i in test]
	return learn.Result(y_true,y_pred,test)