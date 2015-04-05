import itertools
import matplotlib.pyplot as plt
from matplotlib import cm

import glob
import cPickle as pickle
import numpy as np

#methods = ['ola','cla', 'kne','knu','prior','posterior','mcb0', 'mcb1','mcb2','mcb3','mcb4','mcb5','mcb6','mcb7','mcb8','mcb9']
methods = ['ola','cla', 'kne','knu','prior','posterior','mcb0', 'mcb3','mcb6','mcb9']
scoring = ['accuracy', 'f1', 'pr', 'auc-roc']
ref_names = [('Bagging', 'red'), ('AdaBoost','black'), ('Random Forest', 'blue')]

class Results_Combine:
  def __init__(self, dst, v, l, s):
    lmnn_files, lec_files, combine_files = self.get_all_files(dst, v, l, s)
    ref_file = '/'.join([dst.strip('/'), 'refs.pickle'])
    self.refs = pickle.load(open(ref_file))
    self.rr_lmnn = Results(lmnn_files, ref_file, 'lmnn') 
    self.rr_lec = Results(lec_files, ref_file, 'lec') 
    self.rr_combine = Results(combine_files, ref_file, 'lmnn_lec') 
    self.rr_list = [('lmnn', v, self.rr_lmnn), ('lec', l, self.rr_lec), ('combine', s, self.rr_combine)]
  
  def get_all_files(self, dst, v, l, s):
    lmnn_files = glob.glob('/'.join([dst.strip('/'), 'pickles/lmnn_only*v%.2f*.pickle' %v]))
    lec_files =  glob.glob('/'.join([dst.strip('/'), 'pickles/lec_only*l%.2f*.pickle' %l]))
    combine_files =  glob.glob('/'.join([dst.strip('/'), 'pickles/lmnn_lec*v%.2f*l%.2f*step%d*.pickle' %(v, l, s)]))
    return lmnn_files, lec_files, combine_files

  def plot_comp(self, k, n):
    for i, sc in enumerate(scoring):
      des_scores = [self.rr_list[0][2].dict_scores[(k, self.rr_list[0][1], method, n)][0][i] for method in methods]
      comp_scores = [[rr.dict_scores[(k, para, method, n)][1][i] for method in methods] for name, para, rr in self.rr_list]
      ylabel = '%s scores' %sc
      title = '%s scores comparison' %sc
      plot_comp(np.vstack([des_scores, comp_scores]), ['des']+map(lambda x:x[0], self.rr_list), self.refs[:, i], ylabel, title)
      #plt.savefig()

class Results:
  def __init__(self, filenames, ref_file, name):
    self.results = [pickle.load(open(filename)) for filename in filenames]
    self.refs = pickle.load(open(ref_file))
    if 'name' not in self.results[0][0]:
      self.name = name
    else:
      self.name = self.results[0][0]['name']
    self.dict_scores = self.get_dict_scores()

  def get_dict_scores(self):
    dict_scores = {}
    for result in self.results:
      if self.name=='lec':
        paras, acc, stats, ref_acc = result
        para = paras['l']
      elif self.name=='lmnn':
        paras, acc, stats, ref_acc = result
        para = paras['v']
      else:
        paras, acc_stats, ref_acc = result
        para = paras['step']
        bb = acc_stats[0][0].keys()
        acc = dict([(kk, np.vstack([a[0][kk] for a in acc_stats])) for kk in bb]) 
      for method, scores in acc.iteritems():
        #for n in xrange(5, 49, 5):
        for n in xrange(49):
          scores_re = scores[1:,n,:]
          scores_des = scores[0,n,:]
          winner = ((scores_re-scores_des)/(0.01+scores_re.max(axis=0) - scores_des)).sum(axis=1).argmax()
          dict_scores[(paras['k'], para, method, n)] = (scores_des, scores_re[winner])
    return dict_scores

  def plot_bar_scores(self, k, n, para):
    for i, sc in enumerate(scoring):
      des_scores = [self.dict_scores[(k, para, method, n)][0][i] for method in methods]
      winner_scores = [self.dict_scores[(k, para, method, n)][1][i] for method in methods]
      fig, ax = plt.subplots()
      ind = np.arange(len(methods))
      width = 0.4 
      ax.bar(ind, des_scores, width, color='r', label='des')
      ax.bar(ind+width, winner_scores, width, color='y', label='re')
      ax.set_ylabel('%s scores' %sc)
      ax.set_title('%s scores by des methods' %sc)
      ax.set_xticks(ind+width)
      ax.set_xticklabels(methods)
      score_min = np.min(des_scores+winner_scores+self.refs[:, i].tolist())
      score_max = np.max(des_scores+winner_scores+self.refs[:, i].tolist())
      score_diff = score_max - score_min
      if score_min==np.min(self.refs[:, i].tolist()):
        y_axis_bottom = score_min-score_diff*0.1
      else:
        y_axis_bottom = score_min-score_diff*0.5
      ax.set_ylim(\
      bottom=max(0, y_axis_bottom),\
      top=score_max+score_diff*.5)
      for j, (ref_name, ref_color) in enumerate(ref_names):
        ax.axhline(self.refs[j, i], linewidth=2, label=ref_name, color=ref_color)
      ax.legend(loc=9, fontsize='medium', ncol=3)
      plt.show()

  def plot_vl_effect(self, k, n):
    para_list = list(set(map(lambda x:x[1], self.dict_scores.keys())))
    para_list.sort()
    for i, sc in enumerate(scoring):
      para_scores = [[self.dict_scores[(k, pa, method, n)][1][i] for method in methods] for pa in para_list]
      ylabel = '%s scores' %sc
      title = '%s scores with ' %sc
      plot_comp(para_scores, para_list, self.refs[:,i], ylabel, title)

def plot_comp(scores, names, ref, ylabel, title):
  #colors = itertools.cycle(['b', 'g', 'r'])
  colors = itertools.cycle([cm.jet(i/float(len(names))) for i in xrange(len(names))])
  fig, ax = plt.subplots()
  ind = np.arange(len(methods))
  width = 0.85/len(scores)
  for i in xrange(len(names)):
    ax.bar(ind+i*width, scores[i], width, color=colors.next(), label='p=%s'%names[i])
  score_max = max(np.max(scores), np.max(ref))
  score_min = min(np.min(scores), np.min(ref))
  score_diff = score_max-score_min
  if score_min==np.min(ref):
    y_axis_bottom = score_min-score_diff*0.1
  else:
    y_axis_bottom = score_min-score_diff*0.3
  ax.set_ylim(bottom=max(0, y_axis_bottom), top=score_max+score_diff*.4)
  ax.set_xticks(ind+len(names)*width/2)
  ax.set_xticklabels(methods)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  for j, (ref_name, ref_color) in enumerate(ref_names):
    ax.axhline(ref[j], linewidth=2, label=ref_name, color=ref_color, ls='--')
  ax.legend(loc=9, fontsize='medium', ncol=3)
  #plt.savefig(fig_folder+'/%s_%s_%d-%d.png' %(figname, kk, j, min(j+group_size, nk)), format='png')
  #plt.clf()
  plt.show()

if __name__=='__main__':
  rc = Results_Combine('results/bupa_dt_md3/', 0.2, 0.3, 5)

