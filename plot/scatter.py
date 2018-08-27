import matplotlib.pyplot as plt
import numpy as np
import seqs.io,utils 

def scatter_actions(in_path,out_path,indexes=(0,1)):
    read_action=seqs.io.build_action_reader(img_seq=False,as_dict=False)
    actions=read_action(in_path)
    utils.make_dir(out_path)
    for action_i in actions:
        features=action_i.as_features()
        x_i,y_i=features[indexes[0]],features[indexes[1]]
        out_path_i=out_path+'/'+action_i.name
        save_scatter_plot(x_i,y_i,out_path_i)

def save_scatter_plot(x_i,y_i,out_path_i):
    print(out_path_i)
    ax=plt.plot(x_i, y_i, 'o', color='red')
#    ax.get_figure()
    plt.savefig(out_path_i)
    plt.clf()
    plt.close()