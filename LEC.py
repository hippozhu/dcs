import numpy as np
from sklearn.neighbors import NearestNeighbors
import itertools
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
  def __init__(self, X_train, y_train, X_test, y_test, clf, k, l):
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.clf = clf
    self.k = k
    self.l = l
    self.n_train = self.y_train.shape[0]
    self.n_iter_total = 0
    self.sample_weight = np.ones((len(clf.estimators_), y_train.shape[0]))
    self.update_neigh()
    
  def update_neigh(self, M=None):
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
      lec_increment, lec_train = compute_lec_increment(self.neigh_train, pp_train)
      lec_test = compute_lec_test(self.neigh_test, pp_test, pp_train)

      top = np.vstack(rankdata(li, method='max') > (self.n_train-self.l) for li in lec_increment)
      positive_lec_increment = lec_increment>0
      #to_be_adjusted = top & positive_lec_increment
      #to_be_adjusted = positive_lec_increment & self.clf.estimators_samples_
      to_be_adjusted = top & positive_lec_increment & self.clf.estimators_samples_
      to_be_increased = to_be_adjusted & (~pp_train.T)
      to_be_decreased = to_be_adjusted & pp_train.T

      error_rate = 1-np.array([e.score(self.X_train, self.y_train) for e in self.clf.estimators_])
      alpha = np.sqrt(error_rate/(1-error_rate)) # alpha < 1
      for i, (inc, dec) in enumerate(itertools.izip(to_be_increased, to_be_decreased)):
	#sample_weight[i,inc] = sample_weight[i,inc]/alpha[i]
	#sample_weight[i,dec] = sample_weight[i,dec]*alpha[i]
	if inc.sum() > 0:
	  self.sample_weight[i,inc] = self.sample_weight[i,inc]*1.03
	if dec.sum() > 0:
	  self.sample_weight[i,dec] = self.sample_weight[i,dec]/1.03

      print '(%d):%.4f, %.4f, (%d,%d,%d,%d), %.4f@(%d,%d)' %(self.n_iter_total, lec_train.mean(), lec_test.mean(), positive_lec_increment.sum(), to_be_adjusted.sum(), to_be_increased.sum(), to_be_decreased.sum(), des_acc.max(), des_acc.argmax()/des_acc.shape[1], des_acc.argmax()%des_acc.shape[1])
      print lec_increment.max(), lec_increment.min(), lec_increment.mean()
      #print  np.where(des_acc==des_acc.max())
      #print ''
      pprint(des_acc)
      for i, e in enumerate(self.clf.estimators_):
	e.fit(self.X_train[self.clf.estimators_samples_[i]],\
	      self.y_train[self.clf.estimators_samples_[i]],\
	      sample_weight=self.sample_weight[i, self.clf.estimators_samples_[i]])
'''
def local_expertise_enhance(X_train, y_train, X_test, y_test, clf, k):
  n_train = y_train.shape[0]
  sample_weight = np.ones((len(clf.estimators_), y_train.shape[0]))
  yy_train = np.searchsorted(clf.classes_, y_train)
  distances, neighborhoods = NearestNeighbors(k + 1, metric='euclidean', algorithm='brute').fit(X_train).kneighbors(X_train)
  distances = distances[:,1:];neighborhoods = neighborhoods[:,1:]

  l = 5
  acc = []

  max_iter = 100
  for _ in xrange(max_iter):
    acc.append(des_test(X_train, yy_train, X_test, y_test, clf, k))
    preds_train = np.array(map(lambda e:e.predict(X_train), clf.estimators_)).T
    performance = np.array([pt==yt for pt,yt in itertools.izip(preds_train, yy_train)])
    error_rate = 1-np.array([e.score(X_train, yy_train) for e in clf.estimators_])
    lec_increment, lec = compute_lec_increment(preds_train, yy_train, neighborhoods, performance)
    #print ' '.join('{:6.2f}'.format(100*v) for v in acc[-1]) 
    pprint_diff(acc[-1][-1][None,:])
    top = np.vstack(rankdata(li, method='max') > (n_train-l) for li in lec_increment)
    positive_lec_increment = lec_increment>0
    to_be_adjusted = top & positive_lec_increment
    to_be_increased = to_be_adjusted & (~performance.T)
    to_be_decreased = to_be_adjusted & performance.T
    alpha = np.sqrt(error_rate/(1-error_rate)) # alpha < 1
    for i, (inc, dec) in enumerate(itertools.izip(to_be_increased, to_be_decreased)):
      sample_weight[i,inc] = sample_weight[i,inc]/alpha[i]
      sample_weight[i,dec] = sample_weight[i,dec]*alpha[i]

    for i, e in enumerate(clf.estimators_):
      e.fit(X_train[clf.estimators_samples_[i]],\
            yy_train[clf.estimators_samples_[i]],\
	    sample_weight=sample_weight[i, clf.estimators_samples_[i]])

  acc.append(des_test(X_train, yy_train, X_test, y_test, clf, k))
  preds_train = np.array(map(lambda e:e.predict(X_train), clf.estimators_)).T
  performance = np.array([pt==yt for pt,yt in itertools.izip(preds_train, yy_train)])
  lec_increment, lec = compute_lec_increment(preds_train, yy_train, neighborhoods, performance)
  #print ' '.join('{:6.2f}'.format(100*v) for v in acc[-1])
  pprint_diff(acc[-1][-1][None,:])
  
  return np.array(acc)
'''
def des_test1(X_train, y_train, X_test, y_test, clf, k, M=None):
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
  original_acc.fill(accuracy_score(y_test, clf.predict(X_test)))
  return np.hstack((original_acc[:, None], acc))

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
