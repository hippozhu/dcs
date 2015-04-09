import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import itertools
import cPickle as pickle
from collections import Counter
import glob
import cPickle as pickle
import numpy as np
from scipy.stats import rankdata

#methods = ['ola','cla', 'kne','knu','prior','posterior','mcb0', 'mcb1','mcb2','mcb3','mcb4','mcb5','mcb6','mcb7','mcb8','mcb9']
methods = ['ola','cla', 'kne','knu','prior','posterior','mcb0', 'mcb3','mcb6','mcb9']
scoring = ['ACCURACY', 'F1', 'AUC-PR', 'AUC-ROC']
ref_names = [('Bagging', 'green'), ('AdaBoost','black'), ('Random Forest', 'blue')]
excel_colors = ['steelblue', 'indianred', 'darkseagreen', 'slateblue']

def latex_table(ref_scores, des_scores, lec_scores, lmnn_scores, combine_scores):
    lines = []
    selected_scores = [ref_scores, des_scores]
    if lec_scores is not None:
      selected_scores.append(lec_scores)
    if lmnn_scores is not None:
      selected_scores.append(lmnn_scores)
    if combine_scores is not None:
      selected_scores.append(combine_scores)

    all_scores = np.vstack(selected_scores)
    sorted_scores = all_scores.argsort(axis=0)
    top_scores = [s[i] for s, i in itertools.izip(all_scores.T, sorted_scores[-5])]
    max_scores = [s[i] for s, i in itertools.izip(all_scores.T, sorted_scores[-1])]

    lines.append('\multirow{3}{*}{Classis ensembles}')
    for i, scores in enumerate(ref_scores):
      formated_scores = ['%.3f'%score if score < threshold else '\\textbf{%.3f}'%score if score < m else '\\underline{\\textbf{%.3f}}'%score for score, threshold, m in itertools.izip(scores, top_scores, max_scores)]
      lines.append(' & '.join(['&'+ref_names[i][0]] + formated_scores)+'\\\\')
    lines.append('\hline\n')

    lines.append('\multirow{10}{*}{DES alone}')
    for method, scores in itertools.izip(methods, des_scores):
      formated_scores = ['%.3f'%score if score < threshold else '\\textbf{%.3f}'%score if score < m else '\\underline{\\textbf{%.3f}}'%score for score, threshold, m in itertools.izip(scores, top_scores, max_scores)]
      lines.append(' & '.join(['&'+method] + formated_scores)+'\\\\')
    lines.append('\hline\n')
    
    if lec_scores is not None:
      lines.append('\multirow{10}{*}{DES-LEC}')
      for method, scores in itertools.izip(methods, lec_scores):
        formated_scores = ['%.3f'%score if score < threshold else '\\textbf{%.3f}'%score if score < m else '\\underline{\\textbf{%.3f}}'%score for score, threshold, m in itertools.izip(scores, top_scores, max_scores)]
        lines.append(' & '.join(['&'+method+'-lec'] + formated_scores)+'\\\\')
      lines.append('\hline\n')

    if lmnn_scores is not None:
      lines.append('\multirow{10}{*}{DES-LMNN}')
      for method, scores in itertools.izip(methods, lmnn_scores):
        formated_scores = ['%.3f'%score if score < threshold else '\\textbf{%.3f}'%score if score < m else '\\underline{\\textbf{%.3f}}'%score for score, threshold, m in itertools.izip(scores, top_scores, max_scores)]
        lines.append(' & '.join(['&'+method+'-lmnn'] + formated_scores)+'\\\\')
      lines.append('\hline\n')

    if combine_scores is not None:
      lines.append('\multirow{10}{*}{DES-LEC/LMNN}')
      for method, scores in itertools.izip(methods, combine_scores):
        formated_scores = ['%.3f'%score if score < threshold else '\\textbf{%.3f}'%score if score < m else '\\underline{\\textbf{%.3f}}'%score for score, threshold, m in itertools.izip(scores, top_scores, max_scores)]
        lines.append(' & '.join(['&'+method+'-combine'] + formated_scores)+'\\\\')
      lines.append('\hline\n')
    return lines
     
class Results_Combine:
  def __init__(self, dst, v, l, s):
    self.dst = dst
    self.format = 'pdf'
    self.lmnn_files, self.lec_files, self.combine_files = self.get_all_files(dst, v, l, s)
    self.ref_file = '/'.join([dst.strip('/'), 'refs.pickle'])
    self.dataset = dst.split('/')[1]
    self.refs = pickle.load(open(self.ref_file))
    self.rr_lec = Results(self.lec_files, self.ref_file, 'LEC') 
    self.rr_lmnn = Results(self.lmnn_files, self.ref_file, 'LMNN') 
    self.rr_combine = Results(self.combine_files, self.ref_file, 'RE') 
    self.rr_list = [('LEC', l, self.rr_lec.select_kn(), self.rr_lec), ('LMNN', v, self.rr_lmnn.select_kn(), self.rr_lmnn), ('RE', s, self.rr_combine.select_kn(), self.rr_combine)]
    #self.get_performance_scores()
    self.get_performance_scores_best()
  
  def get_all_files(self, dst, v, l, s):
    lmnn_files = glob.glob('/'.join([dst.strip('/'), 'pickles/lmnn_only*v%.2f*.pickle' %v]))
    lec_files =  glob.glob('/'.join([dst.strip('/'), 'pickles/lec_only*l%.2f*.pickle' %l]))
    combine_files =  glob.glob('/'.join([dst.strip('/'), 'pickles/lmnn_lec*v%.2f*l%.2f*step%d*.pickle' %(v, l, s)]))
    return lmnn_files, lec_files, combine_files

  def plot_all(self, k, n):
    for name, para, kn, rr in self.rr_list:
      rr.plot_bar_scores(k, n, para, self.format)
      rr.plot_n_effect(k, para, False, self.format)
      rr.plot_k_effect(n, para, False, self.format)
    self.plot_comp(k, n)

  def plot_all_best(self):
    for name, para, (k, n), rr in self.rr_list:
      rr.plot_bar_scores(k, n, para, self.format)
      rr.plot_n_effect(k, para, False, self.format)
      rr.plot_k_effect(n, para, False, self.format)
    self.plot_comp_best()
  
  def plot_table_all_best(self):
    self.plot_table_best('LEC')
    self.plot_table_best('LMNN')
    self.plot_table_best('RE')
    self.plot_table_best('ALL')

  def plot_comp(self, k, n):
    for i, sc in enumerate(scoring):
      des_scores = [self.rr_list[0][3].dict_scores[(k, self.rr_list[0][1], method, n)][0][i] for method in methods]
      comp_scores = [[rr.dict_scores[(k, para, method, n)][1][i] for method in methods] for name, para, kn, rr in self.rr_list]
      ylabel = '%s scores' %sc
      title = '%s scores comparison' %sc
      prefix = self.dst + '/all_%s'%(sc)
      plot_comp(np.vstack([des_scores, comp_scores]), ['des']+map(lambda x:x[0], self.rr_list), self.refs[:, i], ylabel, title, prefix, self.format)

  def plot_comp_best(self):
    for i, sc in enumerate(scoring):
      ylabel = '%s scores' %sc
      title = '%s scores comparison' %sc
      prefix = self.dst + '/all_%s'%(sc)
      scores = np.vstack([self.des_scores_best[:,i], self.lec_scores_best[:,i], self.lmnn_scores_best[:,i], self.combine_scores_best[:,i]])
      plot_comp(scores, ['DES']+map(lambda x:x[0], self.rr_list), self.refs[:, i], ylabel, title, prefix, self.format)

  def get_performance_scores(self):
    k, n = self.rr_combine.select_kn()
    score_list = []
    for name, para, kn, rr in self.rr_list:
      score_list.append(rr.get_des_re_scores(k, n, para))
    aa = np.array(list(itertools.chain(*score_list)))
    self.des_scores, self.lec_scores, self.lmnn_scores, self.combine_scores = aa[0], aa[1], aa[3], aa[5]
  
  def get_performance_scores_best(self):
    score_list = []
    for name, para, (k, n), rr in self.rr_list:
      score_list.append(rr.get_des_re_scores(k, n, para))
    aa = np.array(list(itertools.chain(*score_list)))
    bb = aa[[0,2,4]]
    self.des_scores_best = bb[np.argmin([ss.sum() for ss in (bb.max(axis=0)-bb)])]
    self.best_lec_des, self.best_lmnn_des, self.best_combine_des = bb
    self.lec_scores_best, self.lmnn_scores_best, self.combine_scores_best = aa[1], aa[3], aa[5]

  def plot_table(self, name):
    if name=='LEC':
      lines = latex_table(self.refs, self.des_scores, self.lec_scores, None, None)
    elif name=='LMNN':
      lines = latex_table(self.refs, self.des_scores, None, self.lmnn_scores, None)
    elif name=='RE':
      lines = latex_table(self.refs, self.des_scores, None, None, self.combine_scores)
    else:
      lines = latex_table(self.refs, self.des_scores, self.lec_scores, self.lmnn_scores, self.combine_scores)
    print '\n'.join(lines)
   
  def plot_table_best(self, name):
    if name=='LEC':
      lines = latex_table(self.refs, self.best_lec_des, self.lec_scores_best, None, None)
    elif name=='LMNN':
      lines = latex_table(self.refs, self.best_lmnn_des, None, self.lmnn_scores_best, None)
    elif name=='RE':
      lines = latex_table(self.refs, self.best_combine_des, None, None, self.combine_scores_best)
    else:
      lines = latex_table(self.refs, self.des_scores_best, self.lec_scores_best, self.lmnn_scores_best, self.combine_scores_best)

    fout = open(self.dst+'/%s.txt'%name, 'w')
    fout.write('\n'.join(lines))
    fout.close()

  def rank_all(self):
    lec_scores = np.vstack((self.refs, self.best_lec_des, self.lec_scores_best))
    lec_rank = np.hstack([rankdata(a, 'min')[:, None] for a in -lec_scores.T])

    lmnn_scores = np.vstack((self.refs, self.best_lmnn_des, self.lmnn_scores_best))
    lmnn_rank = np.hstack([rankdata(a, 'min')[:, None] for a in -lmnn_scores.T])

    combine_scores = np.vstack((self.refs, self.best_combine_des, self.combine_scores_best))
    combine_rank = np.hstack([rankdata(a, 'min')[:, None] for a in -combine_scores.T])

    all_scores = np.vstack((self.refs, self.des_scores_best, self.lec_scores_best, self.lmnn_scores_best, self.combine_scores_best))
    all_rank = np.hstack([rankdata(a, 'min')[:, None] for a in -all_scores.T])
    return lec_rank, lmnn_rank, combine_rank, all_rank

class Results:
  def __init__(self, filenames, ref_file, name):
    self.results = [pickle.load(open(filename)) for filename in filenames]
    self.dst = '/'.join(filenames[0].split('/')[:2])
    self.refs = pickle.load(open(ref_file))
    #if 'name' not in self.results[0][0]:
    if 'name' is not None:
      self.name = name
    else:
      self.name = self.results[0][0]['name']
    self.dict_scores = self.get_dict_scores()

  def get_dict_scores(self):
    dict_scores = {}
    for result in self.results:
      if self.name=='LEC':
        paras, acc, stats, ref_acc = result
        para = paras['l']
      elif self.name=='LMNN':
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

  def best_des(self):
    para_keys = self.dict_scores.keys()
    scores_des = np.vstack([self.dict_scores[kk][0] for kk in para_keys])
    aa = (scores_des.max(axis=0)-scores_des).sum(axis=1).argsort()
    cc = Counter(map(lambda x:(x[0], x[3]), [para_keys[kkk] for kkk in aa[:1000]]))
    selected_k, selected_n = cc.most_common()[0][0]
    return selected_k, selected_n

  def get_des_re_scores(self, k, n, para):
    des_scores = np.array([[self.dict_scores[(k, para, method, n)][0][i] for i in xrange(len(scoring))] for method in methods])
    re_scores = np.array([[self.dict_scores[(k, para, method, n)][1][i] for i in xrange(len(scoring))] for method in methods])
    return des_scores, re_scores

  def plot_bar_scores(self, k, n, para, output_format):
    for i, sc in enumerate(scoring):
      des_scores = [self.dict_scores[(k, para, method, n)][0][i] for method in methods]
      winner_scores = [self.dict_scores[(k, para, method, n)][1][i] for method in methods]
      fig, ax = plt.subplots()
      ind = np.arange(len(methods))
      width = 0.4 
      #width = 0.85/len(2)
      ax.bar(ind, des_scores, width, color='steelblue', label='Original DES')
      ax.bar(ind+width, winner_scores, width, color='indianred', label='%s-boosted DES'%self.name)
      ax.set_ylabel('%s scores' %sc)
      ax.set_xlabel('DES models')
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
      plt.savefig(self.dst+'/perform_%s_%s.%s'%(self.name, sc, output_format), format=output_format)
      plt.close()
      #plt.show()

  def plot_n_effect(self, k, para, include_ref, output_format):
    for i, sc in enumerate(scoring):
      ylabel = '%s scores' %sc
      xlabel = 'Number of base learners selected (n)'
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
      plt.xlabel(xlabel)
      plt.title(title)
      plt.savefig(self.dst+'/nn_%s_%s.%s'%(self.name, sc, output_format), format=output_format)
      plt.close()

  def plot_k_effect(self, n, para, include_ref, output_format):
    for i, sc in enumerate(scoring):
      ylabel = '%s scores' %sc
      xlabel = 'Neighborhood size (k)'
      title = '%s scores with differnt k' %sc
      for method in methods:
        k_score = [self.dict_scores[(k, para, method, n)][1][i] for k in xrange(1, 20, 2)]
        plt.plot(np.arange(1, 20, 2), k_score, linewidth=1, label=method)

      if include_ref:
        for j, (ref_name, ref_color) in enumerate(ref_names):
          plt.axhline(self.refs[j, i], linewidth=3, label=ref_name, color=ref_color, ls=':')

      plt.legend(loc='lower right', fancybox=True, ncol=3, fontsize='small')
      plt.ylabel(ylabel)
      plt.xlabel(xlabel)
      plt.title(title)
      plt.savefig(self.dst+'/kk_%s_%s.%s'%(self.name, sc, output_format), format=output_format)
      plt.close()

  def plot_vl_effect(self, k, n):
    para_list = list(set(map(lambda x:x[1], self.dict_scores.keys())))
    para_list.sort()
    if self.name=='LEC':
      labels = ['l=%s'%p for p in para_list]
    elif self.name=='LMNN':
      labels = ['v=%s'%p for p in para_list]
    else:
      labels = ['s=%s'%p for p in para_list]
    for i, sc in enumerate(scoring):
      para_scores = [[self.dict_scores[(k, pa, method, n)][1][i] for method in methods] for pa in para_list]
      title = 'k=%d, n=%d' %(k, n+1)
      ylabel = '%s scores' %sc
      prefix = self.dst + '/vls_%s_%s'%(self.name, sc)
      plot_comp(para_scores, labels, None, ylabel, title, prefix, 'pdf')

def plot_comp(scores, names, ref, ylabel, title, prefix, output_format):
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
  plt.savefig(prefix+'.%s'%output_format, format=output_format)
  plt.clf()
  #plt.show()

def run_all():
  rc = Results_Combine('results/pima_dt_md3', 0.4, 0.4, 5);rc.plot_all_best();rc.plot_table_all_best()
  rc = Results_Combine('results/bupa_dt_md3', 0.2, 0.3, 5);rc.plot_all_best();rc.plot_table_all_best()
  rc = Results_Combine('results/blood_svm_C1', 0.2, 0.3, 5);rc.plot_all_best();rc.plot_table_all_best()
  rc = Results_Combine('results/breast_dt_md5/', 0.2, 0.1 , 5);rc.plot_all_best();rc.plot_table_all_best()
  rc = Results_Combine('results/australian_dt_md3/', 0.2, 0.3, 5);rc.plot_all_best();rc.plot_table_all_best()
  rc = Results_Combine('results/iono_dt_md5/', 0.2, 0.2, 5);rc.plot_all_best();rc.plot_table_all_best()
  rc = Results_Combine('results/heart_dt_md3/', 0.2, 0.2, 5);rc.plot_all_best();rc.plot_table_all_best()
  rc = Results_Combine('results/sonar_dt_md3/', 0.3, 0.3, 5);rc.plot_all_best();rc.plot_table_all_best()
  rc = Results_Combine('results/heartcleveland_dt_md3/', 0.2, 0.2, 5);rc.plot_all_best();rc.plot_table_all_best()
  rc = Results_Combine('results/iris_dt_md3/', 0.2, 0.1, 5);rc.plot_all_best();rc.plot_table_all_best()
  rc = Results_Combine('results/german_dt_md3/', 0.3, 0.3, 5);rc.plot_all_best();rc.plot_table_all_best()

def get_ranks():
  dsts = [
  ('results/pima_dt_md3', 0.4, 0.4, 5),
  ('results/bupa_dt_md3', 0.2, 0.3, 5),
  ('results/blood_svm_C1', 0.2, 0.3, 5),
  ('results/breast_dt_md5', 0.2, 0.1 , 5),
  ('results/australian_dt_md3', 0.2, 0.3, 5),
  ('results/iono_dt_md5',  0.2, 0.2, 5),
  ('results/heart_dt_md3', 0.2, 0.2, 5), 
  ('results/sonar_dt_md3',0.3, 0.3, 5),
  ('results/heartcleveland_dt_md3', 0.2, 0.2, 5),
  ('results/iris_dt_md3', 0.2, 0.1, 5),
  ('results/german_dt_md3', 0.3, 0.3, 5)
  ]
  ranks = []
  for dst, v, l, s in dsts:
    rc = Results_Combine(dst, v, l, s)
    ranks.append(rc.rank_all())
  pickle.dump(ranks, open('ranks.pickle', 'wb'))

def plot_rank(avg_rank, name):
  colors = itertools.cycle(['steelblue', 'indianred', 'darkseagreen', 'slateblue', 'navajowhite'])
  for i, sc in enumerate(scoring):
    xlabels = ['Ensembles', 'Original DES']
    x_ticks = [1,11,23]
    fig, ax = plt.subplots()

    ax.bar(np.arange(3), avg_rank[:,i][:3], 1, color='steelblue')
    ax.bar(np.arange(5,15), avg_rank[:,i][3:13], 1, color='navajowhite')

    ax.bar(np.arange(17,27), avg_rank[:,i][13:23], 1, color='darkseagreen')
    if name=='ALL':
      xlabels.append('DES-LEC')
    else:
      xlabels.append('DES-%s'%name)

    if avg_rank.shape[0]>23:
      ax.bar(np.arange(29,39), avg_rank[:,i][23:33], 1, color='indianred')
      xlabels.append('DES-LMNN')
      x_ticks.append(35)
      ax.bar(np.arange(41,51), avg_rank[:,i][33:43], 1, color='slateblue')
      xlabels.append('DES-RE')
      x_ticks.append(47)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel('Average rank of %s'%sc)
    ax.set_ylabel('Average rank comparison (%s, %s)'%(name, sc))
    plt.savefig('results/rank_%s_%s.png'%(name, sc), format='png')
    plt.close()

def remove_kne(arr):
  return np.hstack([arr[:2], arr[3:]])

def plot_rank_nokne(avg_rank, name):
  colors = itertools.cycle(['steelblue', 'indianred', 'darkseagreen', 'slateblue', 'navajowhite'])
  for i, sc in enumerate(scoring):
    xlabels = ['Ensembles', 'Original DES']
    x_ticks = [1,9,20]
    fig, ax = plt.subplots()

    ax.bar(np.arange(3), avg_rank[:,i][:3], 1, color='steelblue')
    ax.bar(np.arange(5,14), remove_kne(avg_rank[:,i][3:13]), 1, color='navajowhite')

    ax.bar(np.arange(16,25), remove_kne(avg_rank[:,i][13:23]), 1, color='darkseagreen')
    if name=='ALL':
      xlabels.append('DES-LEC')
    else:
      xlabels.append('DES-%s'%name)

    if avg_rank.shape[0]>23:
      ax.bar(np.arange(27,36), remove_kne(avg_rank[:,i][23:33]), 1, color='indianred')
      xlabels.append('DES-LMNN')
      x_ticks.append(31)
      ax.bar(np.arange(38,47), remove_kne(avg_rank[:,i][33:43]), 1, color='slateblue')
      xlabels.append('DES-RE')
      x_ticks.append(42)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel('Average rank of %s'%sc)

    ax.text(.5,.9,'Average rank comparison (%s, %s)'%(name, sc),
    horizontalalignment='center', fontsize='large',
    transform=ax.transAxes)
    plt.savefig('results/rank_%s_%s.pdf'%(name, sc), format='pdf')
    plt.close()

if __name__=='__main__':
  avg_rank_lec, avg_rank_lmnn, avg_rank_combine, avg_rank_all = pickle.load(open('avg_ranks'))
  plot_rank_nokne(avg_rank_lec, 'LEC')
  plot_rank_nokne(avg_rank_lmnn, 'LMNN')
  plot_rank_nokne(avg_rank_combine, 'RE')
  plot_rank_nokne(avg_rank_all, 'ALL')
