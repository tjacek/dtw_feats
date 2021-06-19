import sys
sys.path.append("..")
from ens import Votes
import exp,systems,pref,files,ens

def exp_desc(paths,out_path=None,clf="LR",info="info"):
    all_systems=[systems.borda_count,
                 systems.bucklin,
                 systems.coombs]
    lines=[get_soft_voting(paths,clf,info)]
    for system_i in all_systems:
        name_i=system_i.__name__
        result_i=pref.ensemble(paths,system=system_i,clf=clf)
        lines.append(get_line(name_i,result_i,info))
    if(out_path):
        files.save_txt(lines,out_path)
    else:
        print(lines)

def get_soft_voting(paths,clf="LR",info="info"):
    result=ens.ensemble(paths["common"],paths["binary"],
		        binary=False,clf=clf)[0]
    return get_line("soft voting",result,info)

def get_line(name_i,result_i,info):
	metrics_i=exp.get_metrics(result_i)
	return "%s,%s,%s" % (name_i,info,metrics_i)

if __name__ == "__main__":
    dataset="ICCCI"
    dir_path="../.." #% dataset
    paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
    paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
    print(paths)
    exp_desc(paths,dataset)