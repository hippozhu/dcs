import cPickle as pickle

from DES import *
from lmnn_pp import *
from LEC import *
from mydata import *
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
  return result

def run_lmnn_alone(lmnns, lecs, max_iter):
  pool = Pool(10)
  results = pool.map(train_lmnn_alone, itertools.izip(lmnns, lecs, itertools.repeat(max_iter)), chunksize=1)
  pool.close()
  pool.join()
  return results 
############# LMNN ######################

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
  return result

def run_lec_alone(lmnns, lecs, max_iter):
  pool = Pool(10)
  results = pool.map(train_lec_alone, itertools.izip(lmnns, lecs, itertools.repeat(max_iter)), chunksize=1)
  pool.close()
  pool.join()
  return results 
############# LEC ######################

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

def lmnn_only(paras, X, y, folds):
  K = range(1,20,2)
  V = [.1, .15, .2]
  for k, v in itertools.product(K, V):
    paras['k'], paras['v'] = k, v
    lmnns = [LMNN_PP(k=paras['k'], alpha=1e-5, mu=paras['mu'], c=paras['c'], v=paras['v']).init_input(X, y, train, test) for train,test in folds]
    lecs = [LEC(clf, paras['k'], paras['l']).init_input(X, y, train, test) for clf, (train, test) in itertools.izip(C, folds)]
    results = run_lmnn_alone(lmnns, lecs, paras['niter_lmnn'])
    acc, stats = process_results(results, y, folds)
    pickle.dump((paras, acc, stats), open('lmnn_alone_k%dv%.2f%d.pickle' %(paras['k'], paras['v'], paras['ns']), 'wb'))
  
def lec_only(paras, X, y, folds):
  C = [BaggingClassifier(base_estimator=SVC(kernel='linear', probability=True, C=5), n_estimators=paras['ns']).fit(X[train], y[train]) for train, test in folds]
  lmnns = [LMNN_PP(k=paras['k'], alpha=1e-5, mu=paras['mu'], c=paras['c'], v=paras['v']).init_input(X, y, train, test) for train,test in folds]
  lecs = [LEC(clf, paras['k'], paras['l']).init_input(X, y, train, test) for clf, (train, test) in itertools.izip(C, folds)]
  results = run_lec_alone(lmnns, lecs, paras['niter_lec'])
  acc, stats = process_results(results, y, folds)
  pickle.dump((paras, acc, stats), open('lec_only_k%dv%.2f%d.pickle' %(paras['k'], paras['v'], paras['ns']), 'wb'))
  return None

if __name__=='__main__':
  nfold=10;nrepeat = 5
  paras = {'mu':0.5,
           'c':1,
           'v':0.3,
           'l':0.05,
           'k':7,
           'ns':10,
           'niter_lmnn':200,
           'niter_lec':20}
  X, y = loadPima()
  folds = pickle.load(open('pima_folds'))
  #folds = [(tr, te) for tr, te in StratifiedKFold(y, nfold)]
  C = [BaggingClassifier(base_estimator=SVC(kernel='linear', probability=True, C=5), n_estimators=paras['ns']).fit(X[train], y[train]) for train, test in folds]
  bagging_acc = np.array([accuracy_score(y[folds[i][1]], C[i].predict(X[folds[i][1]])) for i in xrange(nfold)]).mean()
  for _ in xrange(1):
    #lmnn_only(paras, X, y, folds)
    lec_only(paras, X, y, folds)
