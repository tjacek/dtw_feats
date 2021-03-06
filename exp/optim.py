import sys
sys.path.append("..")
import numpy as np
import reduction,files,feats,rename
import selection,ens,exp

def full_optim(common,binary,step=50,n=40):
	acc,s_feats,s_clf=[],[],[]
	for i in range(n):
		n_feats=i*step
		result,s_clf_i=reduced_selection(common,binary,n_feats)
		acc.append(result.get_acc())
		s_feats.append(n_feats)
		s_clf.append(s_clf_i)
		print(list(zip(acc,s_feats)))
	k=np.argmax(acc)
	return acc[k],s_feats[k],s_clf[k]

def pipe_exp(common,binary,step=100):
	lines=[]
	n_feats=reduce_cross(common,step=step)
	votes=reduction.make_selected_votes(common,binary,
				clf="LR",n_common=n_feats,n_binary=0)
	result=votes.voting(binary=False)
	result.report()
	lines.append(get_info(str(n_feats),common,binary,result))
	result=reduced_selection(common,binary,n_feats)
	result.report()
	lines.append(get_info(str(len(s_clf)),common,binary,result))
	print("\n".join(lines))

def reduced_selection(common,binary,n_feats):
	def helper(common,binary,clf="LR"):
		read=reduction.SepSelected(n_feats,0)
		return ens.make_votes(common,binary,"LR",read)
	s_clf=selection.random_selection(common,binary,1000,27,clf="LR",fun=helper)
	read=reduction.SepSelected(n_feats,0)
	result,votes=ens.ensemble(common,binary,
		clf="LR",binary=False,s_clf=s_clf,read=read)
	return result,s_clf

def get_info(desc,common,binary,result):
	info=exp.exp_info(common,binary,result)
	return "%s,%s,%s,%s" % (desc,"dtw",info[1],info[2])

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
	dir_path="../../dtw_paper/MSR"
	common=files.top_files("%s/common/feats" % dir_path)
	common=["%s/dtw" % common_i for common_i in common]
	binary="%s/sim/feats" % dir_path
	best=full_optim(common,binary)
	print(best)