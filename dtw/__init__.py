import dtw.pairs,files,feats,dtw.knn
import files,learn,dtw.pairs

def dtw_exp(seq_path,out_path):
	files.make_dir(out_path)
	paths=files.get_paths(out_path,["pairs","dtw"])
	dtw.pairs.compute_pairs(seq_path,paths["pairs"])
	compute_feats(paths["pairs"],paths["dtw"])

def compute_feats(pair_path,feat_path):
	pairs=dtw.pairs.read(pair_path)
	train,test=pairs.split()
	to_feats(train,train+test,pairs,feat_path)

def to_feats(train,full,pairs,feat_path):
	dtw_feats=feats.Feats()
	for name_i in full:
		dtw_feats[name_i]=pairs.features(name_i,train)
	dtw_feats.save(feat_path)

def select_feats(pair_path,feat_path,k=3):
	pairs=dtw.pairs.read(pair_path)
	train=pairs.split()[0]
	s_names=dtw.knn.knn_selection(train,pairs,k)
	print(len(train))
	print(len(s_names))
	to_feats(s_names,pairs.keys(),pairs,feat_path)

def knn_clf(pair_path,k=3):
	pairs=dtw.pairs.read(pair_path)
	train,test=pairs.split()
	clf=dtw.knn.KNN(train,pairs)
	y_pred=[clf(name_i,k) for name_i in test]
	y_true=[name_i.get_cat() for name_i in test]
	return learn.Result(y_true,y_pred,test)