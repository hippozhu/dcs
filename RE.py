from lmnn_pp import *
from LEC import *
from sklearn.cross_validation import cross_val_score

def train_lmnn(lmnn_lec):
  lmnn, lec = lmnn_lec
  lmnn.update_input(lec.clf)
  max_iter = 100
  lmnn.fit(max_iter)
  return lmnn

def train_lec(lec_lmnn):
  lec, lmnn = lec_lmnn
  lec.update_input(lmnn.M)
  max_iter = 100
  lec.fit(max_iter)
  return lec

def run_lec(lecs, lmnns):
  pool = Pool(10)
  rr = pool.map(train_lec, itertools.izip(lecs, lmnns), chunksize=1)
  pool.close()
  pool.join()
  return rr

def run_lmnn(lmnns, lecs):
  pool = Pool(10)
  mm = pool.map(train_lmnn, itertools.izip(lmnns, lecs), chunksize=1)
  pool.close()
  pool.join()
  return mm

def des_test_lmnn(lmnn_clf):
  k = 20
  lmnn, lec = lmnn_clf
  if np.array_equal(lmnn.M, np.eye(lmnn.M.shape[0])):
    return des_test(lmnn.X_train, lmnn.y_train, lmnn.X_test, lmnn.y_test, lec.clf, k)
  else:
    if len(lmnn.mm) == 0:
      return [des_test(lmnn.X_train, lmnn.y_train, lmnn.X_test, lmnn.y_test, lec.clf, k, lmnn.M)]
    else:
      return [des_test(lmnn.X_train, lmnn.y_train, lmnn.X_test, lmnn.y_test, lec.clf, k, M) for M in lmnn.mm]

def des_test_lec(lmnn_lec):
  k = 20
  lmnn, lec = lmnn_lec
  return [des_test(lmnn.X_train, lmnn.y_train, lmnn.X_test, lmnn.y_test, clf, k, lmnn.M) for clf in lec.clfs]

def calc_des_lmnn(lmnns, lecs):
  pool = Pool(10)
  acc = pool.map(des_test_lmnn, itertools.izip(lmnns, lecs))
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
  acc = pool.map(des_test_lec, itertools.izip(lmnns, lecs))
  pool.close()
  pool.join()
  return acc
  
if __name__ == '__main__':
  rr = run()
  #pickle.dump(rr, open('Ionok11mu.5c1v.3max500nfold10.pickle', 'wb'))
  #pickle.dump(rr, open('Pimak11mu.5c1v.05max500.pickle', 'wb'))
  pickle.dump(rr, open('Bloodk7mu.5c1v.2.pickle', 'wb'))
