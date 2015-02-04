import sys, time
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

  def ensemble_predict(self, ranking, pct=None, topn=None):
    if pct is not None:
      ensembles = np.greater_equal(ranking, np.minimum(np.percentile(ranking, pct, axis=1), ranking.max(axis=1)).reshape((self.y_test.shape[0],1)))
    else:
      ensembles = np.empty(ranking.shape, dtype=np.bool)
      ensembles.fill(False)
      tops = np.argsort(ranking)[:,-topn:]
      rows = np.hstack([[i]*tops.shape[1] for i in xrange(tops.shape[0])])
      cols = tops.flatten()
      ensembles[rows, cols] = True
    return self.predict(ensembles)

  def predict(self, ensembles, weights = None):
    proba = np.array([preds[ensemble].mean(axis=0) for ensemble, preds in itertools.izip(ensembles, self.preds_proba_test)])
    return self.clf.classes_.take(np.argmax(proba, axis=1), axis=0)

def find_competence_region(X_train, X_test, n_neighbors, M=None):
  if M is not None:
    nn = NearestNeighbors(n_neighbors, algorithm='brute', metric='mahalanobis', VI=M).fit(X_train).kneighbors(X_test, return_distance=False)
  else:
    nn = NearestNeighbors(n_neighbors, metric='euclidean', algorithm='brute').fit(X_train).kneighbors(X_test, return_distance=False)
  return [nn[:, :i] for i in xrange(1, n_neighbors+1, 2)]
  
def des_test(X_train, y_train, X_test, y_test, clf, k, M=None):
  des = DES(X_train, y_train, X_test, y_test, clf)
  des.generate_classifier()
  knn_list = find_competence_region(X_train, X_test, k, M)
  acc = []
  for knn in knn_list:
    pred = []
    knora_eliminate_pred, knora_union_pred = des.knora(knn)
    pred.append(knora_eliminate_pred)
    pred.append(knora_union_pred)
    la_ranking, cla_ranking, lap_ranking = des.dcsla(knn)
    #for p in [70, 50, 30, 10]:
    #  pred.append(des.ensemble_predict(la_ranking, pct=p))
    for n in [1, 2, 5, 10, 20, 30, 50, 70, 90]:
      pred.append(des.ensemble_predict(la_ranking, topn=n))
      #pred.append(des.ensemble_predict(cla_ranking, p))
    acc.append(np.apply_along_axis(accuracy_score, 1, pred, y_test))
  original_acc = np.empty(len(knn_list))
  original_acc.fill(accuracy_score(y_test, clf.predict(X_test)))
  return np.hstack((original_acc[:, None], acc))

def des_test1(X_train, y_train, X_test, y_test, clf, clf1, k, M=None):
  des = DES(X_train, y_train, X_test, y_test, clf)
  des.generate_classifier()
  #des.competence_region(k, M)
  knn_list = find_competence_region(X_train, X_test, k, M)
  acc = []
  for knn in knn_list:
    pred = []
    knora_eliminate_pred, knora_union_pred = des.knora(knn)
    pred.append(knora_eliminate_pred)
    pred.append(knora_union_pred)
    la_ranking, cla_ranking, lap_ranking = des.dcsla(knn)
    #for p in [70, 50, 30, 10]:
    #  pred.append(des.ensemble_predict(la_ranking, pct=p))
    for n in [1, 2, 5, 10, 20, 30, 50, 70, 90]:
      pred.append(des.ensemble_predict(la_ranking, topn=n))
      #pred.append(des.ensemble_predict(cla_ranking, p))
    acc.append(np.apply_along_axis(accuracy_score, 1, pred, y_test))
  original_acc = np.empty(len(knn_list))
  original_acc.fill(accuracy_score(y_test, clf1.predict(X_test)))
  return np.hstack((original_acc[:, None], acc))

