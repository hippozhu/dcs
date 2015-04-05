import sys
import cPickle as pickle

from DES import *
from lmnn_pp import *
from LEC import *
from mydata import *

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron

from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score

from scipy import interp
############# processing ###############
def calc_accuracy_fold(y_true, y_preds):
  return np.array([[accuracy_score(y_true, yy_pred) for yy_pred in y_pred] for y_pred in y_preds])

def calc_scores(y_true, y_preds):
  labels = np.array([0, 1], dtype=np.int32)
  scores =  np.array([[(
  accuracy_score(y_true, labels.take(np.argmax(yy_pred, axis=1))),
  f1_score(y_true, labels.take(np.argmax(yy_pred, axis=1))),
  average_precision_score(y_true, yy_pred[:,1]),
  roc_auc_score(y_true, yy_pred[:,1])
  )
  for yy_pred in y_pred] for y_pred in y_preds])
  return scores

def process_results_method(method_y_trues_results):
  method, y_trues, results = method_y_trues_results
  aa = [[iter_results[0][method] for iter_results in fold_results] for fold_results in results]
  return np.array([calc_scores(y_true_fold, y_preds_fold) for y_preds_fold, y_true_fold in itertools.izip(aa, y_trues)]).mean(axis=0)

def calc_scores_single(y_true, yy_pred):
  labels = np.array([0, 1], dtype=np.int32)
  return accuracy_score(y_true, labels.take(np.argmax(yy_pred, axis=1))), f1_score(y_true, labels.take(np.argmax(yy_pred, axis=1))), average_precision_score(y_true, yy_pred[:,1]), roc_auc_score(y_true, yy_pred[:,1])

def process_results_fold(y_true_result):
  y_true, result = y_true_result
  methods = result[0][0].keys()
  aa = np.array([[iter_results[0][method] for iter_results in result] for method in methods])
  return np.array([[[calc_scores_single(y_true, y_preds) for y_preds in aa_iter] for aa_iter in aa_key] for aa_key in aa])

def process_results(results, y, folds):
  y_trues = [y[te] for _, te in folds]
  methods = results[0][0][0].keys()

  pool = Pool(10)
  fold_scores = pool.map(process_results_fold, itertools.izip(y_trues, results))
  pool.close()
  pool.join()

  acc_dict = dict(itertools.izip(methods, np.mean(fold_scores, axis=0)))
  stats = np.array([[iter_results[1] for iter_results in fold_results] for fold_results in results]).mean(axis=0)
  return acc_dict, stats

def des_evaluate(lmnn_lec):
  lmnn, lec = lmnn_lec
  des = DES_BASE(lmnn, lec)
  preds = des.pred_all()
  return preds

############# LMNN ######################
def train_lmnn_alone(lmnn_lec):
  lmnn, lec, max_iter, is_initial = lmnn_lec
  lmnn.update_input(lec.clf)
  result = []
  if is_initial:
    result.append((des_evaluate((lmnn, lec)), lmnn.report()))
  for _ in xrange(max_iter):
    lmnn.fit(1)
    result.append((des_evaluate((lmnn, lec)), lmnn.report()))
  lmnn.clean()
  return result, lmnn

def run_lmnn_alone(lmnns, lecs, max_iter, is_initial):
  pool = Pool(10)
  results = pool.map(train_lmnn_alone, itertools.izip(lmnns, lecs, itertools.repeat(max_iter), itertools.repeat(is_initial)), chunksize=1)
  pool.close()
  pool.join()
  return results 

def lmnn_only(paras, X, y, folds, C):
  lmnns = [LMNN_PP(k=paras['k'], alpha=1e-5, mu=paras['mu'], c=paras['c'], v=paras['v']).init_input(X, y, train, test) for train,test in folds]
  lecs = [LEC(clf, paras['k'], paras['l']).init_input(X, y, train, test) for clf, (train, test) in itertools.izip(C, folds)]
  results_acc_newlmnn = run_lmnn_alone(lmnns, lecs, paras['niter_lmnn'], True)
  results = map(lambda a:a[0], results_acc_newlmnn)
  acc, stats = process_results(results, y, folds)
  pickle.dump((paras, acc, stats, (bagging_acc, adaboost_acc, randforest_acc)), open('lmnn_only_k%dv%.2f_ns%d.pickle' %(paras['k'], paras['v'], paras['ns']), 'wb'))
  

############# LEC ######################
def train_lec_alone(lmnn_lec):
  lmnn, lec, max_iter, is_initial = lmnn_lec
  lec.update_input(lmnn.M)
  result = []
  if is_initial:
    result.append((des_evaluate((lmnn, lec)), lec.report()))
  for _ in xrange(max_iter):
    lec.fit(1)
    result.append((des_evaluate((lmnn, lec)), lec.report()))
  return result, lec

def run_lec_alone(lmnns, lecs, max_iter, is_initial):
  pool = Pool(10)
  results = pool.map(train_lec_alone, itertools.izip(lmnns, lecs, itertools.repeat(max_iter), itertools.repeat(is_initial)), chunksize=1)
  pool.close()
  pool.join()
  return results 

def lec_only(paras, X, y, folds, C):
  lmnns = [LMNN_PP(k=paras['k'], alpha=1e-5, mu=paras['mu'], c=paras['c'], v=paras['v']).init_input(X, y, train, test) for train,test in folds]
  lecs = [LEC(clf, paras['k'], paras['l']).init_input(X, y, train, test) for clf, (train, test) in itertools.izip(C, folds)]
  results_acc_newlec = run_lec_alone(lmnns, lecs, paras['niter_lec'], True)
  results = map(lambda a:a[0], results_acc_newlec)
  acc, stats = process_results(results, y, folds)
  pickle.dump((paras, acc, stats, (bagging_acc, adaboost_acc, randforest_acc)), open('lec_only_k%dl%.2fns%d.pickle' %(paras['k'], paras['l'], paras['ns']), 'wb'))


############# LMNN_LEC ######################
def lmnn_lec_combine(paras, X, y, folds, C):
  #step_size = [1, 5, 20]
  step_size = [5]
  for step in step_size:
    paras['step'] = step
    niter = paras['niter_lmnn']/step

    lmnns = [LMNN_PP(k=paras['k'], alpha=1e-5, mu=paras['mu'], c=paras['c'], v=paras['v']).init_input(X, y, train, test) for train,test in folds]
    lecs = [LEC(clf, paras['k'], paras['l']).init_input(X, y, train, test) for clf, (train, test) in itertools.izip(C, folds)]

    acc_stats = []

    is_initial = True
    for _ in xrange(niter):
      results_acc_newlmnn = run_lmnn_alone(lmnns, lecs, step, is_initial)
      results = map(lambda a:a[0], results_acc_newlmnn)
      lmnns = map(lambda a:a[1], results_acc_newlmnn)
      acc_stats.append(process_results(results, y, folds))

      is_initial = False

      results_acc_newlec = run_lec_alone(lmnns, lecs, step, is_initial)
      results = map(lambda a:a[0], results_acc_newlec)
      lecs = map(lambda a:a[1], results_acc_newlec)
      acc_stats.append(process_results(results, y, folds))
      
    pickle.dump((paras, acc_stats, (bagging_acc, adaboost_acc, randforest_acc)), open('lmnn_lec_k%dv%.2fl%.2fstep%dns%d.pickle' %(paras['k'], paras['v'], paras['l'], paras['step'], paras['ns']), 'wb'))
    
bagging_acc = .0
adaboost_acc = .0
randforest_acc = .0

if __name__=='__main__':
  nfold=10;nrepeat = 5
  paras = {'mu':0.5,
           'c':1,
           'v':0.2,
           'l':0.05,
           'k':19,
           'ns':50,
           'niter_lmnn':200,
           'niter_lec':200,
           'step':5,
           'niter':40}

  X, y = pickle.load(open('data.pickle'))
  folds = pickle.load(open('folds.pickle'))
  C = pickle.load(open('C.pickle')) 
  paras['ns'] = C[0].n_estimators
  #bagging_acc = np.array([accuracy_score(y[folds[i][1]], C[i].predict(X[folds[i][1]])) for i in xrange(nfold)]).mean()
  #adaboost_acc = cross_val_score(AdaBoostClassifier(base_estimator=C[0].base_estimator, n_estimators=paras['ns']), X, y, cv=folds, n_jobs=10).mean()
  #randforest_acc = cross_val_score(RandomForestClassifier(n_estimators=paras['ns']), X, y, cv=folds, n_jobs=10).mean()

  for _ in xrange(1):
    paras['k'] = int(sys.argv[1])
    if sys.argv[2]=='None':
      paras['l'] = float(sys.argv[3])
      paras['name'] = 'lec'
      lec_only(paras, X, y, folds, C)
    elif sys.argv[3]=='None':
      paras['v'] = float(sys.argv[2])
      paras['name'] = 'lmnn'
      lmnn_only(paras, X, y, folds, C)
    else:
      paras['v'], paras['l'] = float(sys.argv[2]), float(sys.argv[3])
      paras['name'] = 'lmnn_lec'
      lmnn_lec_combine(paras, X, y, folds, C)
