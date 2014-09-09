import sys, time
import numpy as np
import multiprocessing as mp
#import multiprocessing.dummy as mt
import itertools
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from dcs_rank import *
from MyClassifier import *
n_procs = 15

def dcs(X_train, y_train, X_test, y_test, clf):
  clf.fit(X_train, y_train)
  estimators = clf.estimators_
  preds_train = np.array(map(lambda e:e.predict(X_train), estimators)).T
  preds_test = np.array(map(lambda e:e.predict(X_test), estimators)).T

  knora = KNORA(X_train, y_train, preds_train, X_test, y_test, preds_test)
  knora.build_neighborhood(50)
  #selected_ensemble =  

def common_oracles(neighbors_performance_est):
  neighbors, performance_est = neighbors_performance_est
  oracles = None
  for i, neighbor in enumerate(neighbors):
    current_oracles = np.where(performance_est[neighbor])[0]
    if oracles == None:
      oracles = current_oracles
    intersect = np.intersect1d(oracles, current_oracles, True)
    if len(intersect) > 0:
      oracles = intersect
    else:
      break
  if len(oracles)==0:
    oracles = np.array(range(performance_est.shape[1]))
  return i+1, oracles

class KNORA:
  def __init__(self, X_train, y_train, preds_train, X_test, y_test, preds_test):
    self.X_train = X_train
    self.y_train = y_train
    self.preds_train = preds_train
    self.X_test = X_test
    self.y_test = y_test
    self.preds_test = preds_test

  def build_neighborhood(self, n_neighbors):
    self.dist, self.knn = \
    NearestNeighbors(n_neighbors, metric='euclidean', algorithm='brute')\
    .fit(self.X_train)\
    .kneighbors(self.X_test)

    self.dist_est, self.knn_est = \
    NearestNeighbors(n_neighbors, metric='manhattan', algorithm='brute')\
    .fit(self.preds_train)\
    .kneighbors(self.preds_test)

    self.performance_est = np.vstack(pred==y for pred, y in zip(self.preds_train, self.y_train))

  def eliminate(self):
    '''
    pool = mp.Pool(n)
    #pool = mt.Pool(n)
    result = pool.map(common_oracles, itertools.izip(self.knn, itertools.repeat(self.performance_est)), chunksize = 5)
    pool.close()
    pool.join()
    '''
    result = map(common_oracles, itertools.izip(self.knn, itertools.repeat(self.performance_est)))
    return map(lambda x: x[1], result)
  
  #def predict(self, selected, weights = None):
    
if __name__ == '__main__':
  X, y = loadPima()
  clf = MyBaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100)
  folds = [(train, test) for train, test in StratifiedKFold(y, 10)]
  for train, test in folds:
    X_train = X[train];y_train = y[train];X_test = X[test];y_test = y[test]
    clf.fit(X_train, y_train)
    estimators = clf.estimators_
    preds_train = np.array(map(lambda e:e.predict(X_train), estimators)).T;preds_test = np.array(map(lambda e:e.predict(X_test), estimators)).T
    kk = KNORA(X_train, y_train, preds_train, X_test, y_test, preds_test)
    kk.build_neighborhood(int(sys.argv[1]))
    pp = kk.eliminate()
    print accuracy_score(y_test, clf.predict(X_test)), accuracy_score(y_test, clf.predict_selective(X_test, pp))

