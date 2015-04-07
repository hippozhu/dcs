import matplotlib.pyplot as plt
from matplotlib import cm

import itertools
from collections import Counter
import glob
import cPickle as pickle
import numpy as np

#methods = ['ola','cla', 'kne','knu','prior','posterior','mcb0', 'mcb1','mcb2','mcb3','mcb4','mcb5','mcb6','mcb7','mcb8','mcb9']
methods = ['ola','cla', 'kne','knu','prior','posterior','mcb0', 'mcb3','mcb6','mcb9']
scoring = ['ACCURACY', 'F1', 'AUC-PR', 'AUC-ROC']
ref_names = [('Bagging', 'green'), ('AdaBoost','black'), ('Random Forest', 'blue')]
excel_colors = ['steelblue', 'indianred', 'darkseagreen', 'slateblue']

def latex_table(ref_scores, des_scores, lec_scores, lmnn_scores, combine_scores):
  all_scores = np.vstack((ref_scores, des_scores, lec_scores, lmnn_scores, combine_scores))
  top_scores = [s[i] for s, i in itertools.izip(all_scores.T, all_scores.argsort(axis=0)[-5])]

  for i, scores in enumerate(ref_scores):
    print ' & '.join(['&'+ref_names[i][0]] + ['\\textbf{%.3f}'%score if score>=threshold else '%.3f'%score for score, threshold in itertools.izip(scores, top_scores)])+'\\\\'

  print '\hline'

  for method, scores in itertools.izip(methods, des_scores):
    print ' & '.join(['&'+method] + ['\\textbf{%.3f}'%score if score>=threshold else '%.3f'%score for score, threshold in itertools.izip(scores, top_scores)])+'\\\\'
    
  print '\hline'

  for method, scores in itertools.izip(methods, lec_scores):
    print ' & '.join(['&'+method+'-lec'] + ['\\textbf{%.3f}'%score if score>=threshold else '%.3f'%score for score, threshold in itertools.izip(scores, top_scores)])+'\\\\'
    
  print '\hline'

  for method, scores in itertools.izip(methods, lmnn_scores):
    print ' & '.join(['&'+method+'-lmnn'] + ['\\textbf{%.3f}'%score if score>=threshold else '%.3f'%score for score, threshold in itertools.izip(scores, top_scores)])+'\\\\'
    
  print '\hline'

  for method, scores in itertools.izip(methods, combine_scores):
    print ' & '.join(['&'+method+'-combine'] + ['\\textbf{%.3f}'%score if score>=threshold else '%.3f'%score for score, threshold in itertools.izip(scores, top_scores)])+'\\\\'
    
class Results_Combine:
  def __init__(self, dst, v, l, s):
    self.lmnn_files, self.lec_files, self.combine_files = self.get_all_files(dst, v, l, s)
    self.ref_file = '/'.join([dst.strip('/'), 'refs.pickle'])
    self.dataset = dst.split('/')[1]
    self.refs = pickle.load(open(self.ref_file))
    self.rr_lec = Results(self.lec_files, self.ref_file, 'lec') 
    self.rr_lmnn = Results(self.lmnn_files, self.ref_file, 'lmnn') 
    self.rr_combine = Results(self.combine_files, self.ref_file, 'lmnn_lec') 
    self.rr_list = [('lec', v, self.rr_lec), ('lmnn', l, self.rr_lmnn), ('combine', s, self.rr_combine)]
  
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

  def get_performance(self):
    k, n = self.rr_combine.select_kn()
    score_list = []
    for name, para, rr in self.rr_list:
      score_list.append(rr.get_des_re_scores(k, n, para))
    aa = np.array(list(itertools.chain(*score_list)))
    return aa[0], aa[1], aa[3], aa[5]

class Results:
  def __init__(self, filenames, ref_file, name):
    self.results = [pickle.load(open(filename)) for filename in filenames]
    self.dst = '/'.join(filenames[0].split('/')[:2])
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

  def select_kn(self):
    para_keys = self.dict_scores.keys()
    scores = np.vstack([self.dict_scores[kk][1] for kk in para_keys])
    aa = (scores.max(axis=0)-scores).sum(axis=1).argsort()
    cc = Counter(map(lambda x:(x[0], x[3]), [para_keys[kkk] for kkk in aa[:1000]]))
    selected_k, selected_n = cc.most_common()[0][0]
    return selected_k, selected_n

  def get_des_re_scores(self, k, n, para):
    des_scores = np.array([[self.dict_scores[(k, para, method, n)][0][i] for i in xrange(len(scoring))] for method in methods])
    re_scores = np.array([[self.dict_scores[(k, para, method, n)][1][i] for i in xrange(len(scoring))] for method in methods])
    return des_scores, re_scores

  def plot_bar_scores(self, k, n, para):
    for i, sc in enumerate(scoring):
      des_scores = [self.dict_scores[(k, para, method, n)][0][i] for method in methods]
      winner_scores = [self.dict_scores[(k, para, method, n)][1][i] for method in methods]
      fig, ax = plt.subplots()
      ind = np.arange(len(methods))
      width = 0.4 
      #width = 0.85/len(2)
      ax.bar(ind, des_scores, width, color='steelblue', label='des-original')
      ax.bar(ind+width, winner_scores, width, color='indianred', label='des-re')
      ax.set_ylabel('%s scores' %sc)
      ax.set_title('k=%d, n=%d' %(k, n+1))
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
        ax.axhline(self.refs[j, i], linewidth=2, label=ref_name, color=ref_color, ls=':')
      ax.legend(loc=9, fontsize='medium', ncol=2)
      plt.savefig(self.dst+'/perform_%s.pdf'%(sc), format='pdf')
      plt.close()
      #plt.show()

  def plot_n_effect(self, k, para, include_ref):
    for i, sc in enumerate(scoring):
      ylabel = '%s scores' %sc
      title = '%s scores with differnt n' %sc
      for method in methods:
        if method=='kne':
          continue
        n_score = [self.dict_scores[(k, para, method, n)][1][i] for n in xrange(49)]
        plt.plot(np.arange(49), n_score, linewidth=1, label=method)

      if include_ref:
        for j, (ref_name, ref_color) in enumerate(ref_names):
          plt.axhline(self.refs[j, i], linewidth=3, label=ref_name, color=ref_color, ls=':')

      plt.legend(loc='lower right', fancybox=True, ncol=3, fontsize='small')
      plt.ylabel(ylabel)
      plt.title(title)
      plt.savefig(self.dst+'/nn_%s.pdf'%sc, format='pdf')
      plt.close()

  def plot_k_effect(self, n, para, include_ref):
    for i, sc in enumerate(scoring):
      ylabel = '%s scores' %sc
      title = '%s scores with differnt k' %sc
      for method in methods:
        k_score = [self.dict_scores[(k, para, method, n)][1][i] for k in xrange(1, 20, 2)]
        plt.plot(np.arange(1, 20, 2), k_score, linewidth=1, label=method)

      if include_ref:
        for j, (ref_name, ref_color) in enumerate(ref_names):
          plt.axhline(self.refs[j, i], linewidth=3, label=ref_name, color=ref_color, ls=':')

      plt.legend(loc='lower right', fancybox=True, ncol=3, fontsize='small')
      plt.ylabel(ylabel)
      plt.title(title)
      plt.savefig(self.dst+'/kk_%s.pdf'%sc, format='pdf')
      plt.close()

  def plot_vl_effect(self, k, n):
    para_list = list(set(map(lambda x:x[1], self.dict_scores.keys())))
    para_list.sort()
    if self.name=='lec':
      labels = ['l=%s'%p for p in para_list]
    elif self.name=='lmnn':
      labels = ['v=%s'%p for p in para_list]
    else:
      labels = ['s=%s'%p for p in para_list]
    for i, sc in enumerate(scoring):
      para_scores = [[self.dict_scores[(k, pa, method, n)][1][i] for method in methods] for pa in para_list]
      title = 'k=%d, n=%d' %(k, n+1)
      ylabel = '%s scores' %sc
      prefix = self.dst + '/vls_%s_%s'%(self.name, sc)
      plot_comp(para_scores, labels, None, ylabel, title, prefix)

def plot_comp(scores, names, ref, ylabel, title, prefix):
  colors = itertools.cycle(excel_colors[:len(names)])
  #colors = itertools.cycle([cm.jet(i/float(len(names))) for i in xrange(len(names))])
  fig, ax = plt.subplots()
  ind = np.arange(len(methods))
  width = 0.85/len(scores)
  for i in xrange(len(names)):
    ax.bar(ind+i*width, scores[i], width, color=colors.next(), label=names[i])
  if ref is not None: 
    score_max = max(np.max(scores), np.max(ref))
    score_min = min(np.min(scores), np.min(ref))
    score_diff = score_max-score_min
    if score_min==np.min(ref):
      y_axis_bottom = score_min-score_diff*0.1
    else:
      y_axis_bottom = score_min-score_diff*0.3
    for j, (ref_name, ref_color) in enumerate(ref_names):
      ax.axhline(ref[j], linewidth=2, label=ref_name, color=ref_color, ls='--')
  else:
    score_max = np.max(scores)
    score_min = np.min(scores)
    score_diff = score_max-score_min
    y_axis_bottom = score_min-score_diff*0.3
  ax.set_ylim(bottom=max(0, y_axis_bottom), top=score_max+score_diff*.4)
  ax.set_xticks(ind+len(names)*width/2)
  ax.set_xticklabels(methods)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.legend(loc=9, fontsize='medium', ncol=len(names))
  plt.savefig(prefix+'.pdf', format='pdf')
  plt.clf()
  #plt.show()

if __name__=='__main__':
  rc = Results_Combine('results/bupa_dt_md3/', 0.2, 0.3, 5)
