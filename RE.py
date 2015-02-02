from lmnn_pp import *
from LEC import *

def train_lmnn(X_y_tr_te):
  X, y, (train, test), ff = X_y_tr_te
  max_iter = 500
  k = 7;mu=0.5;c=1;v=0.2
  lmnn = LMNN_PP(k=k, alpha=1e-5, mu=mu, c=c, v=v)
  clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),n_estimators=100)
  lmnn.process_input(X, y, train, None, test, clf)
  lmnn.fit(max_iter, ff)
  return lmnn

def run():
  #X, y = loadPima()
  X, y = loadIonosphere()
  nfold = 10
  folds = [(tr, te) for tr, te in StratifiedKFold(y, nfold)]

  pool = Pool(nfold)
  rr = pool.map(train_lmnn,\
  itertools.izip(itertools.repeat(X), itertools.repeat(y), folds, range(nfold)),\
  chunksize=1)
  pool.close()
  pool.join()
  return rr

def des_test_lmnn(lmnn_i):
  k = 50
  lmnn, i = lmnn_i
  if i == -1:
    return des_test(lmnn.X_train, lmnn.y_train, lmnn.X_test, lmnn.y_test, lmnn.clf, k)
  else:
    return des_test(lmnn.X_train, lmnn.y_train, lmnn.X_test, lmnn.y_test, lmnn.clf, k, lmnn.mm[i-1 if len(lmnn.mm)>=i else -1])

def calc_des(lmnns, idx_m):
  pool = Pool(10)
  after = pool.map(des_test_lmnn, itertools.izip(lmnns, idx_m))
  pool.close()
  pool.join()
  return np.mean(after, axis=0)
  #return after

if __name__ == '__main__':
  rr = run()
  #pickle.dump(rr, open('Ionok11mu.5c1v.3max500nfold10.pickle', 'wb'))
  #pickle.dump(rr, open('Pimak11mu.5c1v.05max500.pickle', 'wb'))
  pickle.dump(rr, open('Bloodk7mu.5c1v.2.pickle', 'wb'))
