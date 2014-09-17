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

class DES:
  def __init__(self, X_train, y_train, X_test, y_test, clf, X_val = None, y_val = None):
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.clf = clf
    if y_val is None:
      self.X_cr = X_train
      self.y_cr = y_train
    else:
      self.X_cr = X_val
      self.y_cr = y_val

  def classifier_generation(self):
    self.clf.fit(self.X_train, self.y_train)
    self.estimators = self.clf.estimators_
    self.y_cr_remap = np.searchsorted(self.clf.classes_, self.y_cr)

    self.preds_cr = np.array([e.predict(self.X_cr) for e in self.estimators]).T
    self.preds_proba_cr = np.array([e.predict_proba(self.X_cr) for e in self.estimators]).swapaxes(0,1)
    self.preds_test = np.array([e.predict(self.X_test) for e in self.estimators]).T
    self.preds_proba_test = np.array([e.predict_proba(self.X_test) for e in self.estimators]).swapaxes(0,1)


  def competence_region(self, n_neighbors):
    _, self.knn = NearestNeighbors(n_neighbors, metric='euclidean', algorithm='brute').fit(self.X_cr).kneighbors(self.X_test)

  def knora(self, cr):
    Knora = KNORA(self.preds_cr, self.y_cr_remap, cr)
    return self.predict(Knora.knora_eliminate()), self.predict(Knora.knora_union())

  def predict(self, ensembles, weights = None):
    proba = np.array([preds[ensemble].mean(axis=0) for ensemble, preds in itertools.izip(ensembles, self.preds_proba_test)])
    return self.clf.classes_.take(np.argmax(proba, axis=1), axis=0)

if __name__ == '__main__':
  #X, y = loadPima()
  #X, y = loadBreastCancer()
  #X, y = loadIonosphere()
  X, y = loadYeast()
  #X, y = loadSegmentation()
  #X, y = loadWine()
  #X, y = loadSpam()
  #X, y = loadSonar()
  #X, y = loadBlood()
  clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100)
  acc = []
  for train, test in StratifiedKFold(y, 2):
    X_train = X[train];y_train = y[train]
    X_test = X[test];y_test = y[test]

    '''
    des = DES(X_train, y_train, X_test, y_test, clf)
    '''
    tr, val = [(_, __) for _, __ in StratifiedShuffleSplit(y_train, 1, test_size=.3)][0]
    X_val = X_train[val];y_val= y_train[val]
    X_train = X_train[tr];y_train = y_train[tr]
    des = DES(X_train, y_train, X_test, y_test, clf, X_val, y_val)

    des.classifier_generation()
    des.competence_region(int(sys.argv[1]))
    knora_eliminate_pred, knora_union_pred = des.knora(des.knn)
    clf.fit(X_train, y_train)
    acc.append([accuracy_score(y_test, clf.predict(X_test)),
    accuracy_score(y_test, knora_eliminate_pred),
    accuracy_score(y_test, knora_union_pred)])

  print np.mean(acc, axis=0)
