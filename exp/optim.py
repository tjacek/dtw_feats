import sys
sys.path.append("..")
import numpy as np
import reduction,files,feats,rename

def reduce_cross(in_path,step=50):
	feat_dict=feats.read(in_path)[0]
	n=int(feat_dict.dim()[0]/step)
	acc,dataset=[],[]
	for i in range(n):
		feat_i=reduction.reduce(feat_dict,(i+1)*step)
		dataset.append(feat_i)
		acc.append(rename.cross_validate(feat_i))
		print(acc)
	k=np.argmax(acc)
	return ((k+1)*step)

if __name__ == "__main__":
	dir_path="../../dtw_paper/MSR/"
	common=files.top_files("%s/common/feats" % dir_path)
	common=["%s/dtw" % common_i for common_i in common]
	binary="%s/binary/stats/feats" % dir_path
	n_feats=reduce_cross(common)
	votes=reduction.make_selected_votes(common,binary,clf="LR",n_common=n_feats,n_binary=0)
	result_i=votes.voting(binary=False)
	result_i.report()