from multiprocessing import Pool
import os,sys,itertools,matplotlib
import cPickle as pickle
import numpy as np
import glob
import re

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_pickle(filename):
  if 'lmnn_lec' in filename:
    paras, acc_stats, ref_acc = pickle.load(open(filename))
    bb = acc_stats[0][0].keys()
    step = acc_stats[0][0]['cla'].shape[0]-1
    n_iter = len(acc_stats)
    #acc = dict([(kk, np.vstack([acc_stats[i][0][kk] if i==0 else acc_stats[i][0][kk][1:] for i in xrange(n_iter)])) for kk in bb])
    acc = dict([(kk, np.vstack([acc_stats[i][0][kk] for i in xrange(n_iter)])) for kk in bb])
  else:
    paras, acc, stats, ref_acc = pickle.load(open(filename))

  bb = acc.keys()
  niter, nk, _ = acc.values()[0].shape
  group_size = 10
  for kk in bb:
    for j in xrange(0, nk, group_size):
      #plt.clf()
      plt.axhline(y=ref_acc[0])
      for i in xrange(j, min(j+group_size, nk)):
        current_acc = acc[kk][:,i,0]
        #current_acc = acc[kk][:,i]
        current_acc_max = current_acc.max()
        current_acc_argmax = current_acc.argmax()
        plt.plot(np.arange(0, niter), current_acc, linewidth=1,label='n%d(%.3f@%d)'%(i+1, current_acc_max, current_acc_argmax))
        plt.annotate('%0.3f(@%d,k=%d)' %(current_acc_max, current_acc_argmax, i+1), size='small', weight='bold', xy=(current_acc_argmax, current_acc_max), xytext=(-30,30), xycoords=('data', 'data'), textcoords='offset points', arrowprops=dict(arrowstyle="-|>", fc="w"))
      plt.legend(loc=8, fontsize='small', ncol=5)
      if 'lmnn_only' in filename:
        plt.title('k=%d, v=%.2f' %(paras['k'], paras['v']))
      elif 'lec_only' in filename:
        plt.title('k=%d, l=%.2f' %(paras['k'], paras['l']))
      else:
        #s = int(re.search(r"step(\d+)", filename).groups()[0])
        plt.title('k=%d, v=%.2f, l=%.2f, s=%d' %(paras['k'], paras['v'], paras['l'], paras['step']))
      figname = filename.split('/')[1][:-7]
      plt.savefig(fig_folder+'/%s_%s_%d-%d.png' %(figname, kk, j, min(j+group_size, nk)), format='png')
      plt.clf()

pickle_folder = None
fig_folder = None 

if __name__=='__main__':
  pickle_folder = sys.argv[1]
  fig_folder = sys.argv[2]
  filenames = glob.glob(pickle_folder+'/*.pickle*')
  pool = Pool(int(sys.argv[3]))
  pool.map(plot_pickle, filenames)
  pool.close()
  pool.join()
