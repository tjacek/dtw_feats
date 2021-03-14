import sys
sys.path.append("..")
import files,ens,exp

def binar_exp(common_path,binary_path,out_path):
	lines=[]
	for binary_i in files.top_files(binary_path):
		binary_i="%s/feats" % binary_i
		result_i=ens.ensemble(common_path,binary_i,
						clf="LR",binary=False)[0]
		desc_i=exp.exp_info(common_path,binary_i,result_i)
		desc_i="%s,%s,%s" % desc_i
		print(desc_i)
		lines.append(str(desc_i))
	files.save_txt(lines,out_path)

binary="../../dtw_paper/MSR/binary/1D_CNN_mod"
common="../../dtw_paper/MSR/common/MSR_500"
binar_exp(common,binary,"MSR_dtw.csv")