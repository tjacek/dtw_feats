import dtw.pairs,files,feats

def compute_feats(pair_path,feat_path):
	pairs=dtw.pairs.read(pair_path)
	train=get_train(pairs)
	dtw_feats=feats.Feats()
	for name_i in train:
		dtw_feats[name_i]=pairs.features(name_i,train)
	dtw_feats.save(feat_path)

def get_train(pairs):
	return [name_i for name_i in pairs.keys()
				if(files.person_selector(name_i))]