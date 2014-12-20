import numpy as np
from sklearn.neighbors import NearestNeighbors
import itertools
from scipy.stats import rankdata
from Dcs import *
#from metric_learn import MYLMNN
from lmnn_pp import *

def find_in_neighborhood(neighborhoods, performance):
  n_target = neighborhoods.shape[0]
  in_neigh = [[0] for _ in xrange(n_target)]
  in_neigh = np.zeros((n_target, performance.shape[1]))
  for i, neigh in enumerate(neighborhoods):
    for j in neigh:
      diff = np.array([1 if not agree else -1 for agree in performance[j]==performance[i]])
      in_neigh[j] += diff
  return in_neigh

def compute_lec_increment(preds_train, y_train, neighborhoods, performance):
  k = neighborhoods.shape[1]
  lec = np.array([(performance[i] == performance[neighborhoods[i]]).mean(axis=0) for i in xrange(y_train.shape[0])])
  diff_in_neigh = find_in_neighborhood(neighborhoods, performance)
  return ((1-2*lec)+diff_in_neigh/k).T, lec

def lmnn(X_train, y_train, X_test, y_test, clf, k, v):
  before = des_test(X_train, y_train, X_test, y_test, clf, 5)
  preds_train = np.array(map(lambda e:e.predict(X_train), clf.estimators_)).T
  performance = np.array([pt==yt for pt,yt in itertools.izip(preds_train, y_train)])
  ml = MYLMNN(k=k, max_iter=3000)
  ml.fit(X_train, performance, v)
  after = des_test(X_train, y_train, X_test, y_test, clf, 5, ml.L)
  return before, after

def lmnn1(X_train, y_train, X_test, y_test, clf, k, v):
  before = des_test(X_train, y_train, X_test, y_test, clf, k)
  preds_train = np.array(map(lambda e:e.predict(X_train), clf.estimators_)).T
  performance = np.array([pt==yt for pt,yt in itertools.izip(preds_train, y_train)])
  ml = LMNN_PP(k=k, alpha=0.0001, mu=0.5, c=0.01)
  ml.process_input(X_train, performance)
  ml.fit(v)
  #L = LA.cholesky(ml.M)
  after = des_test(X_train, y_train, X_test, y_test, clf, k, ml.M)
  return before, after

def des_test(X_train, y_train, X_test, y_test, clf, k, M=None):
  p = 70
  des = DES(X_train, y_train, X_test, y_test, clf)
  des.generate_classifier()
  des.competence_region(k, M)
  knora_eliminate_pred, knora_union_pred = des.knora(des.knn)
  la_ranking, cla_ranking, lap_ranking = des.dcsla(des.knn)
  la_pred_max = des.ensemble_predict(la_ranking, 100)
  cla_pred_max = des.ensemble_predict(cla_ranking, 100)
  lap_pred_max = des.ensemble_predict(lap_ranking, 100)
  la_pred = des.ensemble_predict(la_ranking, p)
  cla_pred = des.ensemble_predict(cla_ranking, p)
  lap_pred = des.ensemble_predict(lap_ranking, p)
  return [accuracy_score(y_test, clf.predict(X_test)),
  accuracy_score(y_test, knora_eliminate_pred),
  accuracy_score(y_test, knora_union_pred),
  accuracy_score(y_test, la_pred_max),
  accuracy_score(y_test, cla_pred_max),
  #accuracy_score(y_test, lap_pred_max),
  accuracy_score(y_test, la_pred),
  accuracy_score(y_test, cla_pred),
  #accuracy_score(y_test, lap_pred)
  ]

def local_expertise_enhance(X_train, y_train, X_test, y_test, clf, k):
  n_train = y_train.shape[0]
  sample_weight = np.ones((len(clf.estimators_), y_train.shape[0]))
  yy_train = np.searchsorted(clf.classes_, y_train)
  distances, neighborhoods = NearestNeighbors(k + 1, metric='euclidean', algorithm='brute').fit(X_train).kneighbors(X_train)
  distances = distances[:,1:];neighborhoods = neighborhoods[:,1:]

  l = 5
  acc = []

  max_iter = 20
  for _ in xrange(max_iter):
    acc.append(des_test(X_train, yy_train, X_test, y_test, clf, k))
    preds_train = np.array(map(lambda e:e.predict(X_train), clf.estimators_)).T
    performance = np.array([pt==yt for pt,yt in itertools.izip(preds_train, yy_train)])
    error_rate = 1-np.array([e.score(X_train, yy_train) for e in clf.estimators_])
    lec_increment, lec = compute_lec_increment(preds_train, yy_train, neighborhoods, performance)
    print ' '.join('{:6.2f}'.format(100*v) for v in acc[-1]) 
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
  print ' '.join('{:6.2f}'.format(100*v) for v in acc[-1])
  
  return np.array(acc)

def out_csv(filename, arr):
  fout = open(filename, 'w')
  for row in arr:
    fout.write(','.join(str(a) for a in row) + '\n')
  fout.close()

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
  nfold = 10
  acc = np.zeros((21, 7))
  for train, test in StratifiedKFold(y, nfold):
    print 'fold', i
    i += 1
    X_train = X[train];y_train = y[train];X_test = X[test];y_test = y[test]
    clf.fit(X_train, y_train)
    if len(sys.argv) == 3:
      acc += local_expertise_enhance(X_train, y_train, X_test, y_test, clf, k)
    else:
      b, a = lmnn1(X_train, y_train, X_test, y_test, clf, k, v)
      before.append(b);after.append(a)

  if len(sys.argv) == 3:
    pickle.dump(acc/nfold, open(sys.argv[-1]+'k%d.pickle'%(k), 'w'))
  else:
    b = np.array(before);a = np.array(after)
    pickle.dump((b, a), open(sys.argv[-1]+'k%dv%f.lmnn.pickle'%(k, v), 'w'))
    print b, a
    print b.mean(axis=0), a.mean(axis=0)
    print (a-b).mean(axis=0)

