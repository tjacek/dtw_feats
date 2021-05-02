import sys
sys.path.append("..")
from ens import Votes
import exp,systems,pref,files

def exp_desc(in_path,out_path,info="SVC"):
	votes=pref.read_pref(in_path)
	all_systems=[systems.borda_count,
				systems.bucklin,
				systems.coombs]
	result=votes.voting(False)
	lines=[get_line("soft voting",result,info)]
	for system_i in all_systems:
		name_i=system_i.__name__
		result_i=pref.voting(votes,system_i)
		lines.append(get_line(name_i,result_i,info))
	files.save_txt(lines,out_path)

def get_line(name_i,result_i,info):
	metrics_i=exp.get_metrics(result_i)
	return "%s,%s,%s" % (name_i,info,metrics_i)

if __name__ == "__main__":
	dataset="MSR"
	exp_desc("../SVC/%s" % dataset,dataset)