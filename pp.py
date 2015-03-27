from multiprocessing import Pool
import os,sys,itertools,matplotlib
import cPickle as pickle
import numpy as np
import glob
import re

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_only(picklefolder_filename):
  picklefolder, filename = picklefolder_filename
  figfolder = picklefolder+'_png'
  paras, acc, stats, bagging_acc = pickle.load(open('%s/%s.pickle' %(picklefolder, filename)))
  bb = acc.keys()
  niter, nk = acc.values()[0].shape
  group_size = 10
  for kk in bb:
    for j in xrange(0, nk, group_size):
      plt.clf()
      plt.axhline(y=bagging_acc)
      for i in xrange(j, min(j+group_size, nk)):
        plt.plot(np.arange(0, niter), acc[kk][:,i], linewidth=1,label='k=%d'%(i+1))
        plt.annotate('%0.3f(@%d,k=%d)' %(acc[kk][:,i].max(), acc[kk][:,i].argmax(), i+1), size='small', weight='bold', xy=(acc[kk][:,i].argmax(), acc[kk][:,i].max()), xytext=(-30,30), xycoords=('data', 'data'), textcoords='offset points', arrowprops=dict(arrowstyle="-|>", fc="w"))
      plt.legend(loc=4)
      plt.savefig('%s/%s_%s_%d-%d.png' %(figfolder, filename, kk, j, min(j+group_size, nk)), format='png')

def plot_pickle(filename):
  if 'lmnn_lec' in filename:
    paras, acc_stats, bagging_acc = pickle.load(open(filename))
    bb = acc_stats[0][0].keys()
    step = acc_stats[0][0]['cla'].shape[0]-1
    n_iter = len(acc_stats)
    acc = dict([(kk, np.vstack([acc_stats[i][0][kk] if i==0 else acc_stats[i][0][kk][1:] for i in xrange(n_iter)])) for kk in bb])
  else:
    paras, acc, stats, bagging_acc = pickle.load(open(filename))
# plot_fig(paras, acc, bagging_acc)
#def plot_fig(paras, acc, bagging_acc):
  bb = acc.keys()
  niter, nk = acc.values()[0].shape
  group_size = 10
  for kk in bb:
    for j in xrange(0, nk, group_size):
      #plt.clf()
      plt.axhline(y=bagging_acc)
      for i in xrange(j, min(j+group_size, nk)):
        plt.plot(np.arange(0, niter), acc[kk][:,i], linewidth=1,label='k=%d'%(i+1))
        plt.annotate('%0.3f(@%d,k=%d)' %(acc[kk][:,i].max(), acc[kk][:,i].argmax(), i+1), size='small', weight='bold', xy=(acc[kk][:,i].argmax(), acc[kk][:,i].max()), xytext=(-30,30), xycoords=('data', 'data'), textcoords='offset points', arrowprops=dict(arrowstyle="-|>", fc="w"))
      plt.legend(loc=4)
      if 'lmnn_only' in filename:
        plt.title('k=%d, v=%.2f' %(paras['k'], paras['v']))
      elif 'lec_only' in filename:
        plt.title('k=%d, l=%.2f' %(paras['k'], paras['l']))
      else:
        s = int(re.search(r"step(\d+)", filename).groups()[0])
        plt.title('k=%d, v=%.2f, l=%.2f, s=%d' %(paras['k'], paras['v'], paras['l'], s))
      figname = filename.split('/')[1][:-7]
      plt.savefig('figs/%s_%s_%d-%d.png' %(figname, kk, j, min(j+group_size, nk)), format='png')
      plt.clf()

if __name__=='__main__':
  filenames = glob.glob('pickles/*.pickle')
  pool = Pool(16)
  pool.map(plot_pickle, filenames)
  pool.close()
  pool.join()
  #folder = sys.argv[1]
  #K = range(1,20,2)
  #K = [19] 
  #L = [0.25, 0.3, 0.35, 0.4]
  #filenames = ['lec_only_k%dl%.2fns50' %(k, l) for k, l in itertools.product(K, L)]
  #V = [0.1,0.2,0.3,0.4]
  #filenames = ['lmnn_only_k%dv%.2fns50' %(k, v) for k, v in itertools.product(K, V)]
  #os.mkdir(folder+'_png')
  #pool.map(plot_png, itertools.izip(itertools.repeat(folder), filenames))
