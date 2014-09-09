from sklearn.ensemble import BaggingClassifier
import numpy as np

class MyBaggingClassifier(BaggingClassifier):
  def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
    BaggingClassifier.__init__(self,
                 base_estimator=base_estimator,
                 n_estimators=n_estimators,
                 max_samples=max_samples,
                 max_features=max_features,
                 bootstrap=bootstrap,
                 bootstrap_features=bootstrap_features,
                 oob_score=oob_score,
                 n_jobs=n_jobs,
                 random_state=random_state,
                 verbose=verbose)

  def predict_selective(self, X, selected_estimators_index):
    return self.classes_.take(np.argmax(self.predict_proba_selective(X, selected_estimators_index), axis=1),axis=0)

  def predict_proba_selective(self, X, selected_estimators_index):
    proba = np.vstack([np.mean([self.estimators_[i].predict_proba(x)[0] for i in estimators_index], axis = 0) for x, estimators_index in zip(X, selected_estimators_index)])

    return proba

#def selective_estimators_predict_proba(args):
  
