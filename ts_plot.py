import numpy as np
import matplotlib.pyplot as plt
import exp

@exp.dir_funtion
def make_plots(in_path,out_path):
    d=np.loadtxt(in_path,delimiter=',')
    fig, ax = plt.subplots(figsize=(6, 6))
    x=np.arange(d.shape[0])
    for ts_i in d.T:
        ts_i-=np.mean(ts_i)
        ts_i/=np.std(ts_i)
        ax.plot(x, ts_i)
        print(d.shape[0])
        print(ts_i.shape)	
    plt.show()

in_path="../CZU-MHAD/inert/qyh_a12_t6.mat"
make_plots(in_path,"test")