import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
import itertools
import copy
from scipy.stats import rankdata
from Dcs import *

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
    self.clf_orig = copy.deepcopy(clf)
    self.k = k
    self.l = l
    self.ww = []
    self.clfs = []
    
  def init_input(self, X, y, train, test):
    self.X_train = X[train]
    self.y_train = y[train]
    self.X_test = X[test]
    self.y_test = y[test]
    self.n_train = self.y_train.shape[0]
    self.n_iter_total = 0
    self.sample_weight = np.ones((len(self.clf.estimators_), self.y_train.shape[0]))
    
  def update_input(self, M):
    if M is None:
      self.neigh_train = NearestNeighbors(self.k+1, metric='euclidean', algorithm='brute').fit(self.X_train).kneighbors(self.X_train, return_distance=False)[:,1:]
      self.neigh_test = NearestNeighbors(self.k, metric='euclidean', algorithm='brute').fit(self.X_train).kneighbors(self.X_test, return_distance=False)
    else:
      self.neigh_train = NearestNeighbors(self.k+1, metric='mahalanobis', algorithm='brute', VI=M).fit(self.X_train).kneighbors(self.X_train, return_distance=False)[:,1:]
      self.neigh_test = NearestNeighbors(self.k, metric='mahalanobis', algorithm='brute', VI=M).fit(self.X_train).kneighbors(self.X_test, return_distance=False)

  def fit(self, max_iter):
    for _ in xrange(max_iter):
      self.n_iter_total += 1
      des_acc = des_test(self.X_train, self.y_train, self.X_test, self.y_test, self.clf, self.k)
      preds_train = np.array([e.predict(self.X_train) for e in self.clf.estimators_]).T;pp_train = np.array([pt==yt for pt,yt in itertools.izip(preds_train, self.y_train)])
      preds_test = np.array([e.predict(self.X_test) for e in self.clf.estimators_]).T;pp_test = np.array([pt==yt for pt,yt in itertools.izip(preds_test, self.y_test)])
      self.lec_increment, self.lec_train = compute_lec_increment(self.neigh_train, pp_train)
      self.lec_test = compute_lec_test(self.neigh_test, pp_test, pp_train)

      top = np.vstack(rankdata(li, method='max') > (self.n_train-self.l) for li in self.lec_increment)
      self.positive_lec_increment = self.lec_increment>0
      #self.to_be_adjusted = top & self.positive_lec_increment
      #self.to_be_adjusted = self.positive_lec_increment & self.clf.estimators_samples_
      self.to_be_adjusted = top & self.positive_lec_increment & self.clf.estimators_samples_
      self.to_be_increased = self.to_be_adjusted & (~pp_train.T)
      self.to_be_decreased = self.to_be_adjusted & pp_train.T

      error_rate = 1-np.array([e.score(self.X_train, self.y_train) for e in self.clf.estimators_])
      alpha = np.sqrt(error_rate/(1-error_rate)) # alpha < 1
      for i, (inc, dec) in enumerate(itertools.izip(self.to_be_increased, self.to_be_decreased)):
	#sample_weight[i,inc] = sample_weight[i,inc]/alpha[i]
	#sample_weight[i,dec] = sample_weight[i,dec]*alpha[i]
	if inc.sum() > 0:
	  self.sample_weight[i,inc] = self.sample_weight[i,inc]*1.03
	if dec.sum() > 0:
	  self.sample_weight[i,dec] = self.sample_weight[i,dec]/1.03

      # update individual classifiers by re-training with new sample weights
      for i, e in enumerate(self.clf.estimators_):
	e.fit(self.X_train[self.clf.estimators_samples_[i]],\
	self.y_train[self.clf.estimators_samples_[i]],\
	sample_weight=self.sample_weight[i, self.clf.estimators_samples_[i]])

      if self.n_iter_total % 10 == 0:
        self.report()

  def report(self):
    print '(%d):%.4f,%.4f (%d,%d,%d,%d)' %(self.n_iter_total, self.lec_train.mean(), self.lec_test.mean(), self.positive_lec_increment.sum(), self.to_be_adjusted.sum(), self.to_be_increased.sum(), self.to_be_decreased.sum())
    #, des_acc.max(), des_acc.argmax()/des_acc.shape[1], des_acc.argmax()%des_acc.shape[1]
    #print self.lec_increment.max(), self.lec_increment.min(), self.lec_increment.mean()
    #print  np.where(des_acc==des_acc.max())
    #print ''
    #pprint(des_acc)
    self.ww.append(self.sample_weight.copy())
    self.clfs.append(copy.deepcopy(self.clf))

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
  k = int(sys.argv[1])
  #ff = int(sys.argv[2])
  if len(sys.argv) > 3:
    v = float(sys.argv[2])
  clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),n_estimators=100)
  before = []; after = []
  i = 0
  nfold = 5;val_times = 1.0
  acc = np.zeros((21, 7))
  for train, test in StratifiedKFold(y, nfold):
    print 'fold', i
    i += 1
    X_train = X[train];y_train = y[train];X_test = X[test];y_test = y[test]
    clf.fit(X_train, y_train)
    #if len(sys.argv) == 3:
    acc += local_expertise_enhance(X_train, y_train, X_test, y_test, clf, k)
    '''
    else:
      b, a = lmnn1(X_train, y_train, X_test, y_test, clf, k, v)
      before.append(b);after.append(a)
    '''
  #if len(sys.argv) == 3:
  pickle.dump(acc/nfold, open(sys.argv[-1]+'k%d.pickle'%(k), 'w'))
  '''
  else:
    b = np.array(before);a = np.array(after)
    pickle.dump((b, a), open(sys.argv[-1]+'k%dv%f.lmnn.pickle'%(k, v), 'w'))
    #print b, a
    print ' '.join('{:6.2f}'.format(100*v) for v in b.mean(axis=0))
    print ' '.join('{:6.2f}'.format(100*v) for v in a.mean(axis=0))
    print ' '.join('{:6.2f}'.format(100*v) for v in (a-b).mean(axis=0))
  '''
