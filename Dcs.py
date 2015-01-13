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
