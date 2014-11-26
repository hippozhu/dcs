import numpy as np
from sklearn.neighbors import NearestNeighbors
import itertools

def find_in_neighborhood(neighborhoods, performance):
  n_target = neighborhoods.shape[0]
  in_neigh = [[0] for _ in xrange(n_target)]
  in_neigh = np.zeros((n_target, performance.shape[1]))
  for i, neigh in enumerate(neighborhoods):
    for j in neigh:
      diff = np.array([1 if not agree else -1 for agree in performance[j]==performance[i]])
      in_neigh[j] += diff
  return in_neigh

def compute_lec_increment(preds_train, y_train, neighborhoods):
  performance = np.array([pt==yt for pt,yt in itertools.izip(preds_train, y_train)])
  lec = np.array([(performance[i] == performance[neighborhoods[i]]).mean(axis=0) for i in xrange(y_train.shape[0])]
  diff_in_neigh = find_in_neighborhood(neighborhoods, performance)
  lec_increment = (1-2*lec)+diff_in_neigh/k

def local_expertise_enhance(X_train, y_train, clf, k)
  clf.fit(X_train, y_train)
  preds_train = np.array(map(lambda e:e.predict(X_train), clf.estimators_)).T
  error_rate = 1-np.array([e.score(X_train, y_train) for e in clf.estimators_])
  yy = np.searchsorted(clf.classes_, y_train)
  sample_weight = np.ones((len(clf.estimators_), y_train.shape[0]))
  distances, neighborhoods = NearestNeighbors(k + 1, metric='euclidean', algorithm='brute').fit(X_train).kneighbors(X_train)
  distances = distances[:,1:];neighborhoods = neighborhoods[:,1:]

  for _ in xrange(5):
    lec_increment = compute_lec_increment(preds_train, y_train, neighborhoods)
    Counter(lec_increment.argsort(axis=0)[-5:].flatten())


