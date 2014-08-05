import sys

from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.cross_validation import cross_val_score

from scipy import stats

from mydata import *

random_state = 0

def classify_by_la(local_accuracy, pred):
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

def classify_by_topn_classifier(score, preds_proba, n_classifier):
  selected = score.argsort()[:, -n_classifier:]
  preds_proba_selected = np.vstack([p[s]] for p, s in zip(preds_proba, selected))
  proba = preds_proba_selected.mean(axis=1)
  return proba.argmax(axis=1)

################################################################

def overall_local_accuracy(preds_neigh, Y_neigh):
  acc = [map(lambda pred_neigh: accuracy_score(y_neigh, pred_neigh), pred_neigh_estimators) for pred_neigh_estimators, y_neigh in zip(preds_neigh, Y_neigh)]
  return np.array(acc)

def overall_local_accuracy_proba(preds_neigh_proba, Y_neigh):
  acc = [map(lambda pred_neigh_proba: np.mean([x[y] for x, y in zip(pred_neigh_proba, y_neigh)]), pred_neigh_estimators_proba) for pred_neigh_estimators_proba, y_neigh in zip(preds_neigh_proba, Y_neigh)]
  return np.array(acc)

################################################################

def local_class_accuracy(preds_neigh, YY_neigh, preds_test):
  return np.vstack([class_accuracy(pred_neigh_estimators, y_neigh, pred_test) for pred_neigh_estimators, y_neigh, pred_test in zip(preds_neigh, YY_neigh, preds_test)])

def class_accuracy(pred_neigh_estimators, y_neigh, pred_test):
  cl_idx = map(lambda x: np.where(x[1]==x[0])[0], zip(pred_test, pred_neigh_estimators))
  return [accuracy_score(y_neigh[idx], pred_neigh_estimators[i][idx]) if len(idx)>0 else .0 for i, idx in enumerate(cl_idx)]
  
def local_class_accuracy_proba(preds_neigh, preds_neigh_proba, YY_neigh, preds_test):
  return np.vstack([class_accuracy_proba(pred_neigh_estimators, pred_neigh_estimators_proba, y_neigh, pred_test) for pred_neigh_estimators, pred_neigh_estimators_proba, y_neigh, pred_test in zip(preds_neigh, preds_neigh_proba, YY_neigh, preds_test)])

def class_accuracy_proba(pred_neigh_estimators, pred_neigh_estimators_proba, y_neigh, pred_test):
  cl_idx = map(lambda x: np.where(x[1]==x[0])[0], zip(pred_test, pred_neigh_estimators))
  return [pred_neigh_estimators_proba[i][idx].take(y_neigh[idx]).mean() if len(idx)>0 else .0 for i, idx in enumerate(cl_idx)]
  
################################################################

def dcs_la(clf, preds_test, preds_test_proba, preds_neigh, preds_neigh_proba, YY_neigh, n_classifier):
  ola = overall_local_accuracy(preds_neigh, YY_neigh)
  ola_proba = overall_local_accuracy_proba(preds_neigh_proba, YY_neigh)
  lca = local_class_accuracy(preds_neigh, YY_neigh, preds_test)
  lca_proba = local_class_accuracy_proba(preds_neigh, preds_neigh_proba, YY_neigh, preds_test)

  pred_ola = clf.classes_.take(map(lambda x: classify_by_la(x[0], x[1]), zip(ola, preds_test)))
  pred_lca = clf.classes_.take(map(lambda x: classify_by_la(x[0], x[1]), zip(lca, preds_test)))
  pred_ola_topn = clf.classes_.take(classify_by_topn_classifier(ola, preds_test_proba.swapaxes(0,1), n_classifier))
  pred_lca_topn = clf.classes_.take(classify_by_topn_classifier(lca, preds_test_proba.swapaxes(0,1), n_classifier))
  pred_ola_proba_topn = clf.classes_.take(classify_by_topn_classifier(ola_proba, preds_test_proba.swapaxes(0,1), n_classifier))
  pred_lca_proba_topn = clf.classes_.take(classify_by_topn_classifier(lca_proba, preds_test_proba.swapaxes(0,1), n_classifier))

  return pred_ola, pred_lca, pred_ola_topn, pred_lca_topn, pred_ola_proba_topn, pred_lca_proba_topn

def dcs_cv(X, y, clf, n_neigh, n_classifier, p):
  acc = []
  for train, test in StratifiedKFold(y, 3, random_state=random_state):
    X_train = X[train];y_train = y[train];X_test = X[test];y_test = y[test]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    estimators = clf.estimators_
    neigh = NearestNeighbors(n_neigh, metric='euclidean', algorithm='brute')
    neigh.fit(X_train)
    preds_train = np.array(map(lambda e:e.predict(X_train), estimators))
    preds_train_proba = np.array(map(lambda e:e.predict_proba(X_train), estimators))
    preds_test = np.array(map(lambda e:e.predict(X_test), estimators)).T
    preds_test_proba = np.array(map(lambda e:e.predict_proba(X_test), estimators))
    dist, knn = neigh.kneighbors(X_test)
    preds_neigh = np.array(map(lambda nn: preds_train[:,nn], knn))
    preds_neigh_proba = np.array(map(lambda nn: preds_train_proba[:,nn,:], knn))
    Y_neigh = np.array(map(lambda nn: y_train[nn], knn))
    dict_aa = dict((clf.classes_[i], i) for i in xrange(clf.classes_.shape[0]))
    to_class_index = np.vectorize(lambda x: dict_aa[x])
    YY_neigh = to_class_index(Y_neigh)

    preds = dcs_la(clf, preds_test, preds_test_proba, preds_neigh, preds_neigh_proba, YY_neigh, n_classifier)
    acc.append([accuracy_score(y_test, pred) for pred in (y_pred,) + preds])

  mean_acc = np.mean(acc, axis=0)
  print ' '.join('{:.3f}'.format(v) for v in mean_acc)
  return mean_acc

if __name__ == '__main__':
  n_estimators = int(sys.argv[1])
  depth = int(sys.argv[2])

  n_neigh = int(sys.argv[3])
  n_classifier = int(sys.argv[4])

  p = float(sys.argv[5])

  X, y = loadPima()
  #X, y = loadBreastCancer()
  #X, y = loadIonosphere()
  clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=n_estimators)
  #clf = AdaBoostClassifier(n_estimators=n_estimators)
  #clf = BaggingClassifier(n_estimators=n_estimators)
  print np.mean([dcs_cv(X, y, clf, n_neigh, n_classifier, p) for _ in xrange(10)], axis=0)
