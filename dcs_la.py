import sys

from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.cross_validation import cross_val_score

from scipy import stats

from mydata import *

random_state = 0

def classify_by_ola(local_accuracy, pred):
  sorted_acc = np.sort(np.unique(local_accuracy))[::-1]
  for a in sorted_acc:
    acc_indices = np.where(local_accuracy==a)[0]
    val, count = stats.mode(pred[acc_indices])
    if count[0] > acc_indices.shape[0]/2:
      return val[0]
  print 'tie'
  return stats.mode(pred)[0][0]

def classify_by_ola_proba(local_accuracy, pred):
  sorted_acc = np.sort(np.unique(local_accuracy))[::-1]
  for a in sorted_acc:
    acc_indices = np.where(local_accuracy==a)[0] 
    val, count = stats.mode(pred[acc_indices])
    if count[0] > acc_indices.shape[0]/2:
      return val[0]
  print 'tie'
  return stats.mode(pred)[0][0]

def classify_no_last(la, preds_proba, n_last):
  selected = la.argsort()[:, n_last:]
  preds_proba_selected = np.vstack([p[s]] for p, s in zip(preds_proba, selected))
  proba = preds_proba_selected.mean(axis=1)
  return proba.argmax(axis=1)

def overall_local_accuracy(preds_neigh, Y_neigh):
  acc = [map(lambda pred_neigh: accuracy_score(y_neigh, pred_neigh), pred_neigh_estimators) for pred_neigh_estimators, y_neigh in zip(preds_neigh, Y_neigh)]
  return np.array(acc)

def overall_local_accuracy_proba(preds_neigh_proba, Y_neigh):
  acc = [map(lambda pred_neigh_proba: np.mean([x[y] for x, y in zip(pred_neigh_proba, y_neigh)]), pred_neigh_estimators_proba) for pred_neigh_estimators_proba, y_neigh in zip(preds_neigh_proba, Y_neigh)]
  return np.array(acc)

#def local_class_accuracy():
  
def dcs_la(X_train, y_train, X_test, y_test, clf, n_neigh):
  estimators = clf.estimators_
  neigh = NearestNeighbors(n_neigh, metric='euclidean', algorithm='brute')
  neigh.fit(X_train)
  preds_train = np.array(map(lambda e:e.predict(X_train), estimators))
  preds_train_proba = np.array(map(lambda e:e.predict_proba(X_train), estimators))
  preds_test = np.array(map(lambda e:e.predict(X_test), estimators))
  preds_test_proba = np.array(map(lambda e:e.predict_proba(X_test), estimators))
  dist, knn = neigh.kneighbors(X_test)
  preds_neigh = np.array(map(lambda nn: preds_train[:,nn], knn))
  preds_neigh_proba = np.array(map(lambda nn: preds_train_proba[:,nn,:], knn))
  Y_neigh = np.array(map(lambda nn: y_train[nn], knn))
  dict_aa = dict((clf.classes_[i], i) for i in xrange(clf.classes_.shape[0]))
  to_class_index = np.vectorize(lambda x: dict_aa[x])
  YY_neigh = to_class_index(Y_neigh)

  ola = overall_local_accuracy(preds_neigh, YY_neigh)
  ola_proba = overall_local_accuracy_proba(preds_neigh_proba, YY_neigh)
  #best = ola.argmax(axis=1)
  #best_proba = ola_proba.argmax(axis=1)
  #pred_dcs = [clf.classes_[pred[idx]] for idx, pred in zip(best, preds_test.T)]
  #pred_dcs_proba = [clf.classes_[pred[idx]] for idx, pred in zip(best_proba, preds_test.T)]
  pred_dcs = clf.classes_.take(map(lambda x: classify_by_ola(x[0], x[1]), zip(ola, preds_test.T)))
  pred_dcs_proba = clf.classes_.take(map(lambda x: classify_by_ola_proba(x[0], x[1]), zip(ola_proba, preds_test.T)))
  pred_dcs_nolast = clf.classes_.take(classify_no_last(ola, preds_test_proba.swapaxes(0,1), 30))
  return pred_dcs, pred_dcs_proba, pred_dcs_nolast

def dcs_cv(X, y, clf, n_neigh, p):
  acc_clf = []
  acc_dcs = []
  acc_dcs_proba = []
  acc_dcs_new = []
  acc_dcs_nolast = []
  for train, test in StratifiedKFold(y, 3, random_state=random_state):
    X_train = X[train];y_train = y[train];X_test = X[test];y_test = y[test]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    pred_dcs, pred_dcs_proba, pred_dcs_nolast = dcs_la(X_train, y_train, X_test, y_test, clf, n_neigh)

    new_pred = y_pred.copy()
    preds_test = np.array(map(lambda e:e.predict(X_test), clf.estimators_))
    pred_avg = preds_test.mean(axis=0)
    mid = np.array(filter(lambda i: pred_avg[i]>=p and pred_avg[i]<=1-p, range(test.shape[0]))) 
    #print mid.shape[0]
    new_pred[mid] = pred_dcs[mid]

    acc_clf.append(accuracy_score(y_test, y_pred))
    acc_dcs.append(accuracy_score(y_test, pred_dcs))
    acc_dcs_proba.append(accuracy_score(y_test, pred_dcs_proba))
    acc_dcs_new.append(accuracy_score(y_test, new_pred))
    acc_dcs_nolast.append(accuracy_score(y_test, pred_dcs_nolast))
  #print np.mean(acc_clf), acc_clf
  #print np.mean(acc_dcs), acc_dcs
  #print np.mean(acc_dcs_proba), acc_dcs_proba
  #print np.mean(acc_dcs_new), acc_dcs_new
  print '%.3f, %.3f, %.3f, %.3f, %.3f' %(np.mean(acc_clf), np.mean(acc_dcs), np.mean(acc_dcs_proba), np.mean(acc_dcs_new), np.mean(acc_dcs_nolast))
  return np.mean(acc_clf), np.mean(acc_dcs), np.mean(acc_dcs_proba), np.mean(acc_dcs_new), np.mean(acc_dcs_nolast)

'''
def dcs(X, y):
  folds = [(train, test) for train, test in StratifiedKFold(y, 3, random_state=random_state)]
  train, test = folds[0]
  X_train = X[train];y_train = y[train];X_test = X[test];y_test = y[test]
  clf = BaggingClassifier()
  clf.fit(X_train, y_train)
  dcs_la(X_train, y_train, X_test, y_test, clf.estimators_)
'''

if __name__ == '__main__':
  n_neigh = int(sys.argv[1])
  n_estimators = int(sys.argv[2])
  depth = int(sys.argv[3])
  p = float(sys.argv[4])
  X, y = loadPima()
  #X, y = loadBreastCancer()
  #X, y = loadIonosphere()
  #clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=n_estimators)
  clf = AdaBoostClassifier(n_estimators=n_estimators)
  #clf = BaggingClassifier(n_estimators=n_estimators)
  print np.mean([dcs_cv(X, y, clf, n_neigh, p) for _ in xrange(10)], axis=0)
