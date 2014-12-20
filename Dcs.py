import sys, time
import multiprocessing as mp
import itertools
import cPickle as pickle

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

  def generate_classifier(self):
    self.estimators = self.clf.estimators_
    self.y_cr_remap = np.searchsorted(self.clf.classes_, self.y_cr)

    self.preds_cr = np.array([e.predict(self.X_cr) for e in self.estimators]).T
    self.preds_proba_cr = np.array([e.predict_proba(self.X_cr) for e in self.estimators]).swapaxes(0,1)
    self.preds_test = np.array([e.predict(self.X_test) for e in self.estimators]).T
    self.preds_proba_test = np.array([e.predict_proba(self.X_test) for e in self.estimators]).swapaxes(0,1)

  def competence_region(self, n_neighbors, M=None):
    if M is not None:
      #X_cr_t = L.dot(self.X_cr.T).T
      #X_test_t = L.dot(self.X_test.T).T
      _, self.knn = NearestNeighbors(n_neighbors, algorithm='brute', metric='mahalanobis', VI=M).fit(self.X_cr).kneighbors(self.X_test)
    else:
      _, self.knn = NearestNeighbors(n_neighbors, metric='euclidean', algorithm='brute').fit(self.X_cr).kneighbors(self.X_test)

  def knora(self, cr):
    Knora = KNORA(self.preds_cr, self.y_cr_remap, cr)
    return self.predict(Knora.knora_eliminate()), self.predict(Knora.knora_union())

  def dcsla(self, cr):
    Dcsla = DCSLA(self.preds_cr, self.preds_proba_cr, self.y_cr_remap, self.preds_test, cr)
    la = Dcsla.local_accuracy()
    cla = Dcsla.class_local_accuracy(self.clf.classes_)
    lap = Dcsla.local_accuracy_proba()
    return la, cla, lap

  def ensemble_predict(self, ranking, p):
    ensembles = np.greater_equal(ranking, np.minimum(np.percentile(ranking, p, axis=1), ranking.max(axis=1)).reshape((self.y_test.shape[0],1)))
    return self.predict(ensembles)

  def predict(self, ensembles, weights = None):
    proba = np.array([preds[ensemble].mean(axis=0) for ensemble, preds in itertools.izip(ensembles, self.preds_proba_test)])
    return self.clf.classes_.take(np.argmax(proba, axis=1), axis=0)

if __name__ == '__main__':
  myFuncs = {
  'pima': loadPima,
  'breast': loadBreastCancer,
  'iono': loadIonosphere,
  'yeast': loadYeast,
  'seg': loadSegmentation,
  'wine': loadWine,
  'spam': loadSpam,
  'sonar': loadSonar,
  'blood': loadBlood,
  }
  X, y = myFuncs[sys.argv[-1]]()
  print X.shape
  clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100)
  acc = []
  for train, test in StratifiedKFold(y, 5):
    X_train = X[train];y_train = y[train]
    X_test = X[test];y_test = y[test]

    clf.fit(X_train, y_train)
    des = DES(X_train, y_train, X_test, y_test, clf)
    '''
    tr, val = [(_, __) for _, __ in StratifiedShuffleSplit(y_train, 1, test_size=.5)][0]
    X_val = X_train[val];y_val= y_train[val]
    X_train = X_train[tr];y_train = y_train[tr]
    des = DES(X_train, y_train, X_test, y_test, clf, X_val, y_val)
    '''

    des.generate_classifier()
    des.competence_region(int(sys.argv[1]))
    knora_eliminate_pred, knora_union_pred = des.knora(des.knn)
    la_ranking, cla_ranking, lap_ranking = des.dcsla(des.knn)
    la_pred_max = des.ensemble_predict(la_ranking, 100)
    cla_pred_max = des.ensemble_predict(cla_ranking, 100)
    lap_pred_max = des.ensemble_predict(lap_ranking, 100)
    la_pred = des.ensemble_predict(la_ranking, int(sys.argv[2]))
    cla_pred = des.ensemble_predict(cla_ranking, int(sys.argv[2]))
    lap_pred = des.ensemble_predict(lap_ranking, int(sys.argv[2]))
    #clf.fit(X_train, y_train)
    acc.append([accuracy_score(y_test, clf.predict(X_test)),
    accuracy_score(y_test, knora_eliminate_pred),
    accuracy_score(y_test, knora_union_pred),
    accuracy_score(y_test, la_pred_max),
    accuracy_score(y_test, cla_pred_max),
    accuracy_score(y_test, lap_pred_max),
    accuracy_score(y_test, la_pred),
    accuracy_score(y_test, cla_pred),
    accuracy_score(y_test, lap_pred)])

  mean_acc = np.mean(acc, axis=0)
  print ' '.join('{:6.2f}'.format(100*v) for v in mean_acc)
  print ' '.join('{:6.2f}'.format(100*v) for v in mean_acc-mean_acc[0])
