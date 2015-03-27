import sys
import cPickle as pickle

from DES import *
from lmnn_pp import *
from LEC import *
from mydata import *
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

############# processing ###############
def calc_accuracy_fold(y_true, y_preds):
  return np.array([[accuracy_score(y_true, yy_pred) for yy_pred in y_pred] for y_pred in y_preds])

def process_results(results, y, folds):
  y_true = [y[te] for _, te in folds]
  methods = results[0][0][0].keys()
  acc_dict = {}
  for method in methods:
    aa = [[iter_results[0][method] for iter_results in fold_results] for fold_results in results]
    acc_dict[method] = np.array([calc_accuracy_fold(y_true_fold, y_preds_fold) for y_preds_fold, y_true_fold in itertools.izip(aa, y_true)]).mean(axis=0)
  stats = np.array([[iter_results[1] for iter_results in fold_results] for fold_results in results]).mean(axis=0)
  return acc_dict, stats

def des_evaluate(lmnn_lec):
  lmnn, lec = lmnn_lec
  des = DES_BASE(lmnn, lec)
  preds = des.pred_all()
  return preds



############# LMNN ######################
def train_lmnn_alone(lmnn_lec):
  lmnn, lec, max_iter = lmnn_lec
  lmnn.update_input(lec.clf)
  result = []
  for _ in xrange(max_iter):
    result.append((des_evaluate((lmnn, lec)), lmnn.report()))
    lmnn.fit(1)
  result.append((des_evaluate((lmnn, lec)), lmnn.report()))
  lmnn.clean()
  return result, lmnn

def run_lmnn_alone(lmnns, lecs, max_iter):
  pool = Pool(10)
  results = pool.map(train_lmnn_alone, itertools.izip(lmnns, lecs, itertools.repeat(max_iter)), chunksize=1)
  pool.close()
  pool.join()
  return results 

def lmnn_only(paras, X, y, folds, C):
  lmnns = [LMNN_PP(k=paras['k'], alpha=1e-5, mu=paras['mu'], c=paras['c'], v=paras['v']).init_input(X, y, train, test) for train,test in folds]
  lecs = [LEC(clf, paras['k'], paras['l']).init_input(X, y, train, test) for clf, (train, test) in itertools.izip(C, folds)]
  results_acc_newlmnn = run_lmnn_alone(lmnns, lecs, paras['niter_lmnn'])
  results = map(lambda a:a[0], results_acc_newlmnn)
  acc, stats = process_results(results, y, folds)
  pickle.dump((paras, acc, stats, bagging_acc), open('lmnn_only_k%dv%.2f_ns%d.pickle' %(paras['k'], paras['v'], paras['ns']), 'wb'))
  

############# LEC ######################
def train_lec_alone(lmnn_lec):
  lmnn, lec, max_iter = lmnn_lec
  lec.update_input(lmnn.M)
  result = []
  for _ in xrange(max_iter):
    result.append((des_evaluate((lmnn, lec)), lec.report()))
    lec.fit(1)
  result.append((des_evaluate((lmnn, lec)), lec.report()))
  #lmnn.clean()
  return result, lec

def run_lec_alone(lmnns, lecs, max_iter):
  pool = Pool(10)
  results = pool.map(train_lec_alone, itertools.izip(lmnns, lecs, itertools.repeat(max_iter)), chunksize=1)
  pool.close()
  pool.join()
  return results 

def lec_only(paras, X, y, folds, C):
  lmnns = [LMNN_PP(k=paras['k'], alpha=1e-5, mu=paras['mu'], c=paras['c'], v=paras['v']).init_input(X, y, train, test) for train,test in folds]
  lecs = [LEC(clf, paras['k'], paras['l']).init_input(X, y, train, test) for clf, (train, test) in itertools.izip(C, folds)]
  results_acc_newlec = run_lec_alone(lmnns, lecs, paras['niter_lec'])
  results = map(lambda a:a[0], results_acc_newlec)
  acc, stats = process_results(results, y, folds)
  pickle.dump((paras, acc, stats, bagging_acc), open('lec_only_k%dl%.2fns%d.pickle' %(paras['k'], paras['l'], paras['ns']), 'wb'))


############# LMNN_LEC ######################
def lmnn_lec_combine(paras, X, y, folds, C):
  step_size = [1,5,10]
  for step in step_size:
    paras['step'] = step
    niter = paras['niter_lmnn']/step

    lmnns = [LMNN_PP(k=paras['k'], alpha=1e-5, mu=paras['mu'], c=paras['c'], v=paras['v']).init_input(X, y, train, test) for train,test in folds]
    lecs = [LEC(clf, paras['k'], paras['l']).init_input(X, y, train, test) for clf, (train, test) in itertools.izip(C, folds)]

    acc_stats = []

    for _ in xrange(niter):
      results_acc_newlmnn = run_lmnn_alone(lmnns, lecs, step)
      results = map(lambda a:a[0], results_acc_newlmnn)
      lmnns = map(lambda a:a[1], results_acc_newlmnn)
      acc_stats.append(process_results(results, y, folds))

      results_acc_newlec = run_lec_alone(lmnns, lecs, step)
      results = map(lambda a:a[0], results_acc_newlec)
      lecs = map(lambda a:a[1], results_acc_newlec)
      acc_stats.append(process_results(results, y, folds))

    pickle.dump((paras, acc_stats, bagging_acc), open('lmnn_lec_k%dv%.2fl%.2fstep%dns%d.pickle' %(paras['k'], paras['v'], paras['l'], paras['step'], paras['ns']), 'wb'))
    
bagging_acc = .0

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
  #paras['ns'] = int(sys.argv[1])
  X, y = loadPima()

  #folds = [(tr, te) for tr, te in StratifiedKFold(y, nfold)]
  folds = pickle.load(open('pima_folds'))

  #C = [BaggingClassifier(base_estimator=SVC(kernel='linear', probability=True, C=5), n_estimators=paras['ns']).fit(X[train], y[train]) for train, test in folds]
  #C = [BaggingClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=paras['ns']).fit(X[train], y[train]) for train, test in folds]
  C = pickle.load(open('pima_C_dt_md3_ns50.pickle'))
  #C = pickle.load(open('pima_C_svm_C5_ns50.pickle'))

  bagging_acc = np.array([accuracy_score(y[folds[i][1]], C[i].predict(X[folds[i][1]])) for i in xrange(nfold)]).mean()

  for _ in xrange(1):
    paras['k'], paras['v'], paras['l'] = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    lmnn_only(paras, X, y, folds, C)
    lec_only(paras, X, y, folds, C)
    lmnn_lec_combine(paras, X, y, folds, C)

    '''
    K = range(1,20,2)
    V = [0.1,0.2,0.3,0.4]
    L = [float(sys.argv[1])]

    for k, l in itertools.product(K, L):
      paras['k'], paras['l'] = k, l
      lec_only(paras, X, y, folds, C)
    #lmnn_lec_combine(paras, X, y, folds, C)
    '''
