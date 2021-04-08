import sys
sys.path.append("..")
import files,ens,exp,selection,reduction

class Exp(object):
	def __init__(self,fun=None):
		if(fun is None):
			fun=simple_ens
		self.fun=fun

	def __call__(self,common_path,binary_path,info):
		lines=[]
		for binary_i in files.top_files(binary_path):
			id_i=binary_i.split('/')[-1]
			path_i="%s/feats" % binary_i
			desc_i=self.fun(common_path,path_i)
			desc_i="%s,%s,%s" % (info,id_i,desc_i)
			print(desc_i)
			lines.append(str(desc_i))
		return lines

def simple_ens(common,binary):
	result_i= ens.ensemble(common,binary,
						clf="LR",binary=False)[0]
	desc_i=exp.exp_info(common,binary,result_i)
	return desc_i[-1]

def select_ens(common,binary):
	best_set=selection.random_selection(common,binary,1000,27)
	result_i=ens.ensemble(common,binary, s_clf=best_set,
						clf="LR",binary=False)[0]
	desc_i=exp.exp_info(common,binary,result_i)
	return "%d,%s" % (len(best_set),desc_i[-1])

def ts_exp(common,binary,out_path):
	exp=Exp()
	lines=exp(None,binary,"A")
	lines+=exp(common,binary,"F")
	exp=Exp(select_ens)
	lines+=exp(common,binary,"H")
	files.save_txt(lines,out_path)

def make_reduced_dataset(common,out_path,n_feats=350):
	common=files.get_paths(common)
	dataset=ens.read_dataset(common,None)[0]
	dataset.norm()
	redu_data=reduction.reduce(dataset,n_feats)
	redu_data.save(out_path)

dataset="MSR"
binary="../../dtw_paper/%s/binary/1D_CNN" % dataset
in_common="../../dtw_paper/%s/common/feats" % dataset
out_common="../../dtw_paper/%s/common/%s_350" % (dataset,dataset)
make_reduced_dataset(in_common,out_common,n_feats=350)
#ts_exp(common,binary,"%s" % dataset)