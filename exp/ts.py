import sys
sys.path.append("..")
import files,ens,exp,selection

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
			result_i=self.fun(common_path,path_i)
			desc_i=exp.exp_info(common_path,path_i,result_i)
			desc_i="%s,%s,%s" % desc_i
			desc_i="%s,%s,%s" % (info,id_i,desc_i)
			print(desc_i)
			lines.append(str(desc_i))
		return lines

def simple_ens(common_path,path_i):
	return ens.ensemble(common_path,path_i,
						clf="LR",binary=False)[0]

def select_ens(common,binary):
	best_set=selection.random_selection(common,binary,1000,20)
	return ens.ensemble(common,binary, s_clf=best_set,
						clf="LR",binary=False)[0]

def ts_exp(common,binary,out_path):
	exp=Exp()
	lines=exp(None,binary,"A")
	lines+=exp(common,binary,"E")
	exp=Exp(select_ens)
	lines+=exp(common,binary,"H")
	files.save_txt(lines,out_path)

dataset="MHAD"
binary="../../dtw_paper/%s/binary/1D_CNN" % dataset
common="../../dtw_paper/%s/common/%s_500" % (dataset,dataset)
ts_exp(common,binary,"%s" % dataset)