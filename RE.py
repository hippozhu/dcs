from lmnn_pp import *
from LEC import *
from DES import *
from sklearn.cross_validation import cross_val_score

def train_lmnn(lmnn_lec):
  lmnn, lec, max_iter = lmnn_lec
  lmnn.update_input(lec.clf)
  lmnn.fit(max_iter)
  return lmnn

def train_lec(lec_lmnn):
  lec, lmnn, max_iter = lec_lmnn
  lec.update_input(lmnn.M)
  lec.fit(max_iter)
  return lec

def run_lec(lecs, lmnns, max_iter):
  pool = Pool(10)
  rr = pool.map(train_lec, itertools.izip(lecs, lmnns, itertools.repeat(max_iter)), chunksize=1)
  pool.close()
  pool.join()
  return rr

def run_lmnn(lmnns, lecs, max_iter):
  pool = Pool(10)
  mm = pool.map(train_lmnn, itertools.izip(lmnns, lecs, itertools.repeat(max_iter)), chunksize=1)
  pool.close()
  pool.join()
  return mm

def des_test_lmnn(lmnn_clf):
  k = 50
  lmnn, lec = lmnn_clf
  if np.array_equal(lmnn.M, np.eye(lmnn.M.shape[0])):
    return des_test(lmnn.X_train, lmnn.y_train, lmnn.X_test, lmnn.y_test, lec.clf, k)
  else:
    if len(lmnn.mm) == 0:
      return [des_test(lmnn.X_train, lmnn.y_train, lmnn.X_test, lmnn.y_test, lec.clf, k, lmnn.M)]
    else:
      return [des_test(lmnn.X_train, lmnn.y_train, lmnn.X_test, lmnn.y_test, lec.clf, k, M) for M in lmnn.mm]

def des_test_lec(lmnn_lec):
  k = 50
  lmnn, lec = lmnn_lec
  return [des_test(lmnn.X_train, lmnn.y_train, lmnn.X_test, lmnn.y_test, clf, k, lmnn.M) for clf in lec.clfs]

def calc_des_lmnn(lmnns, lecs):
  pool = Pool(10)
  acc = pool.map(des_test_lmnn, itertools.izip(lmnns, lecs), chunksize=1)
  pool.close()
  pool.join()
  return acc

def fill_acc(acc):
  max_len = np.max([len(a) for a in acc])
  filled_acc = []
  for a in acc:
    a_len = len(a)
    if a_len < max_len:
      filled_acc.append(a + [a[-1]]*(max_len-a_len))
    else:
      filled_acc.append(a)
  return filled_acc

def calc_des_lec(lmnns, lecs):
  pool = Pool(10)
  acc = pool.map(des_test_lec, itertools.izip(lmnns, lecs), chunksize=1)
  pool.close()
  pool.join()
  return acc
  
def des_test_all(lmnn_lec):
  k = 50
  lmnn, lec = lmnn_lec
  return des_test(lmnn.X_train, lmnn.y_train, lmnn.X_test, lmnn.y_test, lec.clf, k, lmnn.M)

def calc_des(lmnns, lecs):
  pool = Pool(10)
  acc = pool.map(des_test_all, itertools.izip(lmnns, lecs), chunksize=1)
  pool.close()
  pool.join()
  return acc

def alternate(lmnns, n_iter_lmnn, lecs, n_iter_lec, n_iter):
  acc = []
  for _ in xrange(n_iter):
    lmnns = run_lmnn(lmnns, lecs, n_iter_lmnn)
    acc.append(calc_des(lmnns, lecs))
    lecs = run_lec(lecs, lmnns, n_iter_lec)
    acc.append(calc_des(lmnns, lecs))
  return acc, lmnns, lecs

if __name__ == '__main__':
  folds = pickle.load(open('folds.pickle', 'rb'))
  C = pickle.load(open('C.pickle', 'rb'))
  acc_0 = pickle.load(open('acc_0.pickle', 'rb'))
  '''
  folds, C, acc_0 = pickle.load(open('Ionosphere.pickle', 'rb'))
  '''

  mu=0.5;c=1
  k, v, l = 20, 0.4, 0.05
  #ss_lmnn, ss_lec, n_step = 10, 10, 100
  ss_lmnn, ss_lec, n_step = 20, 20, 50 

  X, y = loadPima()
  #X, y = loadIonosphere()
  lmnns = [LMNN_PP(k=k, alpha=1e-5, mu=mu, c=c, v=v).init_input(X, y, train, test) for train,test in folds]
  lecs = [LEC(clf, k, l).init_input(X, y, train, test) for clf, (train, test) in itertools.izip(C, folds)]

  #acc, lmnns, lecs = alternate(lmnns, 40, lecs, 20, 1)
  acc1, lmnns1, lecs1 = alternate(lmnns, ss_lmnn, lecs, ss_lec, n_step)

