import numpy as np
from sklearn.neighbors import NearestNeighbors
import itertools
import copy
from scipy.stats import rankdata
#from Dcs import *

def find_in_neighborhood(neighborhoods, performance):
  n_target = neighborhoods.shape[0]
  #in_neigh = [[0] for _ in xrange(n_target)]
  in_neigh = np.zeros((n_target, performance.shape[1]))
  for i, neigh in enumerate(neighborhoods):
    for j in neigh:
      diff = np.array([1 if not agree else -1 for agree in performance[j]==performance[i]])
      in_neigh[j] += diff
  return in_neigh

def compute_lec_increment(neigh_train, pp_train):
  n, k = neigh_train.shape
  lec = np.array([(pp_train[i] == pp_train[neigh_train[i]]).mean(axis=0) for i in xrange(n)])
  diff_in_neigh = find_in_neighborhood(neigh_train, pp_train)
  return ((1-2*lec)+diff_in_neigh/k).T, lec.T

def compute_lec_test(neigh_test, pp_test, pp_train):
  n, k = neigh_test.shape
  lec = np.array([(pp_test[i] == pp_train[neigh_test[i]]).mean(axis=0) for i in xrange(n)])
  return lec.T

class LEC:
  def __init__(self, clf, k, l):
    self.clf = clf
    #self.clf_orig = copy.deepcopy(clf)
    self.k = k
    self.l = l
    self.ww = []
    self.clfs = []
    self.stats = []
    
  def init_input(self, X, y, train, test):
    self.X_train = X[train]
    self.y_train = y[train]
    self.X_test = X[test]
    self.y_test = y[test]
    self.n_train = self.y_train.shape[0]
    self.n_iter_total = 0
    self.sample_weight = np.ones((len(self.clf.estimators_), self.y_train.shape[0]))
    return self
    
  def update_input(self, M):
    if M is None:
      self.neigh_train = NearestNeighbors(self.k+1, metric='euclidean', algorithm='brute').fit(self.X_train).kneighbors(self.X_train, return_distance=False)[:,1:]
      self.neigh_test = NearestNeighbors(self.k, metric='euclidean', algorithm='brute').fit(self.X_train).kneighbors(self.X_test, return_distance=False)
    else:
      self.neigh_train = NearestNeighbors(self.k+1, metric='mahalanobis', algorithm='brute', VI=M).fit(self.X_train).kneighbors(self.X_train, return_distance=False)[:,1:]
      self.neigh_test = NearestNeighbors(self.k, metric='mahalanobis', algorithm='brute', VI=M).fit(self.X_train).kneighbors(self.X_test, return_distance=False)

    self.update_lec()

  def update_lec(self):
      preds_train = np.array([e.predict(self.X_train) for e in self.clf.estimators_]).T;pp_train = np.array([pt==yt for pt,yt in itertools.izip(preds_train, self.y_train)])
      preds_test = np.array([e.predict(self.X_test) for e in self.clf.estimators_]).T;pp_test = np.array([pt==yt for pt,yt in itertools.izip(preds_test, self.y_test)])
      self.lec_increment, self.lec_train = compute_lec_increment(self.neigh_train, pp_train)
      self.lec_test = compute_lec_test(self.neigh_test, pp_test, pp_train)
      preds_proba_train = np.array([e.predict_proba(self.X_train) for e in self.clf.estimators_])
      closeness = np.abs(preds_proba_train[:,:,0]-preds_proba_train[:,:,1])
      close = closeness<self.l
      self.positive_lec_increment = self.lec_increment>0
      #top = np.vstack(rankdata(li, method='max') > (self.n_train-self.l) for li in self.lec_increment)
      #self.to_be_adjusted = self.positive_lec_increment & self.clf.estimators_samples_
      self.to_be_adjusted = close & self.positive_lec_increment & self.clf.estimators_samples_
      self.to_be_increased = self.to_be_adjusted & (~pp_train.T)
      self.to_be_decreased = self.to_be_adjusted & pp_train.T

  def fit(self, max_iter):
    for _ in xrange(max_iter):
      self.n_iter_total += 1

      error_rate = 1-np.array([e.score(self.X_train, self.y_train) for e in self.clf.estimators_])
      alpha = np.sqrt(error_rate/(1-error_rate)) # alpha < 1
      for i, (inc, dec) in enumerate(itertools.izip(self.to_be_increased, self.to_be_decreased)):
	if inc.sum() > 0:
	  #sample_weight[i,inc] = sample_weight[i,inc]/alpha[i]
	  self.sample_weight[i,inc] = self.sample_weight[i,inc]*1.001
	if dec.sum() > 0:
	  #sample_weight[i,dec] = sample_weight[i,dec]*alpha[i]
	  self.sample_weight[i,dec] = self.sample_weight[i,dec]/1.001

      # update individual classifiers by re-training with new sample weights
      for i, e in enumerate(self.clf.estimators_):
	e.fit(self.X_train[self.clf.estimators_samples_[i]],\
	self.y_train[self.clf.estimators_samples_[i]],\
	sample_weight=self.sample_weight[i, self.clf.estimators_samples_[i]])

      self.update_lec()

  def report(self):
    if self.n_iter_total % 10 == 0:
      print '(%d):%.4f,%.4f (%d,%d,%d,%d)' %(self.n_iter_total, self.lec_train.mean(), self.lec_test.mean(), self.positive_lec_increment.sum(), self.to_be_adjusted.sum(), self.to_be_increased.sum(), self.to_be_decreased.sum())
    return self.lec_train.mean(), self.lec_test.mean(), self.positive_lec_increment.sum(), self.to_be_adjusted.sum(), self.to_be_increased.sum(), self.to_be_decreased.sum()

