import sys, time
import multiprocessing as mp
import itertools

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit

from mydata import *
from KNORA import *

def oracles_intersect(neighborhood_oracles):
  common_oracles = np.all(neighborhood_oracles, axis=0)
  if common_oracles.any():
    return common_oracles
  common_oracles = np.ones(neighborhood_oracles.shape[1], dtype=bool)
  for i, no in enumerate(neighborhood_oracles):
    next_common_oracles = common_oracles & no
    if not np.any(next_common_oracles):
      break
    else:
      common_oracles = next_common_oracles
  return common_oracles

def oracles_union(neighborhood_oracles):
  common_oracles = np.any(neighborhood_oracles, axis=0)
  if common_oracles.any():
    return common_oracles
  else:
    return np.ones(neighborhood_oracles.shape[1], dtype=bool)

class DES:
  def __init__(self, X_train, y_train, X_test, y_test, clf):
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.clf = clf

  def preprocessing(self, n_neighbors):
    train, val = [(_, __) for _, __ in StratifiedShuffleSplit(y_train, 1, test_size=.3)][0]
    self.X_val = self.X_train[val];self.y_val = self.y_train[val]
    self.X_train = self.X_train[train];self.y_train = self.y_train[train]
    self.clf.fit(self.X_train, self.y_train)
    self.preds_train = np.array(map(lambda e:e.predict(self.X_train), self.clf.estimators_)).T
    self.preds_val = np.array(map(lambda e:e.predict(self.X_val), self.clf.estimators_)).T
    self.preds_proba_test = np.array([e.predict_proba(self.X_test) for e in self.clf.estimators_]).swapaxes(0,1)

    #_, self.knn = NearestNeighbors(n_neighbors, metric='euclidean', algorithm='brute').fit(self.X_train).kneighbors(self.X_test)
    _, self.knn = NearestNeighbors(n_neighbors, metric='euclidean', algorithm='brute').fit(self.X_val).kneighbors(self.X_test)

    #self.oracles = np.vstack(pred==y for pred, y in zip(self.preds_train, self.y_train))
    self.oracles = np.vstack(pred==y for pred, y in zip(self.preds_val, self.y_val))
  
  def knora(self):
    Knora = KNORA(self.oracles, self.knn)
    ensembles_eliminate = Knora.knora_eliminate()
    ensembles_union = Knora.knora_union()
    return self.predict(ensembles_eliminate), self.predict(ensembles_union)

  def predict(self, ensembles, weights = None):
    proba = np.array([preds[ensemble].mean(axis=0) for ensemble, preds in itertools.izip(ensembles, self.preds_proba_test)])
    return self.clf.classes_.take(np.argmax(proba, axis=1), axis=0)

if __name__ == '__main__':
  #X, y = loadPima()
  #X, y = loadBreastCancer()
  #X, y = loadIonosphere()
  #X, y = loadYeast()
  #X, y = loadSegmentation()
  #X, y = loadWine()
  #X, y = loadSpam()
  #X, y = loadSonar()
  X, y = loadBlood()
  clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100)
  acc = []
  for train, test in StratifiedKFold(y, 2):
    X_train = X[train];y_train = y[train];X_test = X[test];y_test = y[test]
    kk = DES(X_train, y_train, X_test, y_test, clf)
    kk.preprocessing(int(sys.argv[1]))
    knora_eliminate_pred, knora_union_pred = kk.knora()
    clf.fit(X_train, y_train)
    acc.append([accuracy_score(y_test, clf.predict(X_test)),
    accuracy_score(y_test, knora_eliminate_pred),
    accuracy_score(y_test, knora_union_pred)])

  print np.mean(acc, axis=0)
