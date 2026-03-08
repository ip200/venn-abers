import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from venn_abers import VennAbersCalibrator

class DummyEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.kwargs_received = None
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y, **kwargs):
        self.kwargs_received = kwargs
        return self
        
    def predict_proba(self, X):
        return np.ones((len(X), 2)) * 0.5
        
def test_kwargs_passing():
    X = np.ones((10, 2))
    y = np.array([0, 1] * 5)
    
    cv = VennAbersCalibrator(estimator=DummyEstimator(), inductive=True, cal_size=0.2)
    cv.fit(X, y, eval_set="mock_eval_set", early_stopping_rounds=10)
    
    underlying_cv = cv.va_calibrator_.multiclass_va_estimators[0]
    assert underlying_cv.estimators_[0].kwargs_received['eval_set'] == "mock_eval_set"
    assert underlying_cv.estimators_[0].kwargs_received['early_stopping_rounds'] == 10
