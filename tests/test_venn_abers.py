import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from venn_abers import VennAbers, VennAbersCV, VennAbersMultiClass, VennAbersCalibrator, VennAbersRegressor

@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=200, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def multiclass_data():
    X, y = make_classification(n_samples=200, n_classes=3, n_informative=5, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_venn_abers_manual(binary_data):
    X_train, X_test, y_train, y_test = binary_data
    X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    
    clf = GaussianNB()
    clf.fit(X_train_proper, y_train_proper)
    
    p_cal = clf.predict_proba(X_cal)
    p_test = clf.predict_proba(X_test)
    
    va = VennAbers()
    va.fit(p_cal, y_cal)
    
    p_prime, p0_p1 = va.predict_proba(p_test)
    
    assert p_prime.shape == (len(X_test), 2)
    assert p0_p1.shape == (len(X_test), 2)
    assert np.allclose(np.sum(p_prime, axis=1), 1.0)

def test_venn_abers_calibrator_ivap_binary(binary_data):
    X_train, X_test, y_train, y_test = binary_data
    
    clf = GaussianNB()
    va = VennAbersCalibrator(estimator=clf, inductive=True, cal_size=0.2, random_state=42)
    va.fit(X_train, y_train)
    
    p_prime = va.predict_proba(X_test)
    y_pred = va.predict(X_test)
    
    assert p_prime.shape == (len(X_test), 2)
    assert np.allclose(np.sum(p_prime, axis=1), 1.0)
    assert y_pred.shape == (len(X_test), 2)

def test_venn_abers_calibrator_cvap_binary(binary_data):
    X_train, X_test, y_train, y_test = binary_data
    
    clf = GaussianNB()
    va = VennAbersCalibrator(estimator=clf, inductive=False, n_splits=3, random_state=42)
    va.fit(X_train, y_train)
    
    p_prime = va.predict_proba(X_test)
    y_pred = va.predict(X_test)
    
    assert p_prime.shape == (len(X_test), 2)
    assert np.allclose(np.sum(p_prime, axis=1), 1.0)

def test_venn_abers_calibrator_ivap_multiclass(multiclass_data):
    X_train, X_test, y_train, y_test = multiclass_data
    
    clf = GaussianNB()
    va = VennAbersCalibrator(estimator=clf, inductive=True, cal_size=0.2, random_state=42)
    va.fit(X_train, y_train)
    
    p_prime = va.predict_proba(X_test)
    y_pred = va.predict(X_test)
    
    assert p_prime.shape == (len(X_test), 3)
    assert np.allclose(np.sum(p_prime, axis=1), 1.0)

def test_venn_abers_calibrator_cvap_multiclass(multiclass_data):
    X_train, X_test, y_train, y_test = multiclass_data
    
    clf = GaussianNB()
    va = VennAbersCalibrator(estimator=clf, inductive=False, n_splits=3, random_state=42)
    va.fit(X_train, y_train)
    
    p_prime = va.predict_proba(X_test)
    y_pred = va.predict(X_test)
    
    assert p_prime.shape == (len(X_test), 3)
    assert np.allclose(np.sum(p_prime, axis=1), 1.0)

def test_venn_abers_regressor_ivap(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    
    reg = LinearRegression()
    va_reg = VennAbersRegressor(estimator=reg, inductive=True, cal_size=0.2, random_state=42)
    va_reg.fit(X_train, y_train)
    
    mid, interval = va_reg.predict(X_test)
    
    assert mid.shape == (len(X_test),)
    assert interval.shape == (len(X_test), 2)
    assert np.all(interval[:, 0] <= interval[:, 1])

def test_venn_abers_regressor_cvap(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    
    reg = LinearRegression()
    # Adding epsilon to not have edge cases fail
    va_reg = VennAbersRegressor(estimator=reg, inductive=False, n_splits=3, random_state=42)
    va_reg.fit(X_train, y_train, m=1)
    
    mid, interval = va_reg.predict(X_test)
    
    assert mid.shape == (len(X_test),)
    assert interval.shape == (len(X_test), 2)
    assert np.all(interval[:, 0] <= interval[:, 1])

def test_venn_abers_calibrator_manual_multiclass(multiclass_data):
    X_train, X_test, y_train, y_test = multiclass_data
    X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    
    clf = GaussianNB()
    clf.fit(X_train_proper, y_train_proper)
    
    p_cal = clf.predict_proba(X_cal)
    p_test = clf.predict_proba(X_test)
    
    va = VennAbersCalibrator()
    p_prime, p0_p1 = va.predict_proba(p_cal=p_cal, y_cal=y_cal, p_test=p_test, p0_p1_output=True)
    
    assert p_prime.shape == (len(X_test), 3)
    assert np.allclose(np.sum(p_prime, axis=1), 1.0)
    assert len(p0_p1) > 0

def test_venn_abers_cv_manual(binary_data):
    X_train, X_test, y_train, y_test = binary_data
    
    clf = GaussianNB()
    va_cv = VennAbersCV(estimator=clf, inductive=False, n_splits=3)
    va_cv.fit(X_train, y_train)
    
    p_prime = va_cv.predict_proba(X_test)
    
    assert p_prime.shape == (len(X_test), 2)
    assert np.allclose(np.sum(p_prime, axis=1), 1.0)

def test_venn_abers_multiclass_manual(multiclass_data):
    X_train, X_test, y_train, y_test = multiclass_data
    clf = GaussianNB()
    va_mc = VennAbersMultiClass(estimator=clf, inductive=True, cal_size=0.2, random_state=42)
    va_mc.fit(X_train, y_train)
    p_prime = va_mc.predict_proba(X_test)
    
    assert p_prime.shape == (len(X_test), 3)
    assert np.allclose(np.sum(p_prime, axis=1), 1.0)


def test_venn_abers_cv_estimators_cloned(binary_data):
    X_train, X_test, y_train, y_test = binary_data
    
    clf = GaussianNB()
    va_cv = VennAbersCV(estimator=clf, inductive=False, n_splits=3)
    va_cv.fit(X_train, y_train)
    
    # Check that estimators are independent objects (different Python ids)
    estimator_ids = [id(est) for est in va_cv.estimators_]
    assert len(estimator_ids) == len(set(estimator_ids)), "Estimators in self.estimators_ must be distinct objects"


def test_venn_abers_cv_predict_interval_ensemble():
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    
    # Generate noisy regression data so fold estimators have different coefficients
    X, y = make_regression(n_samples=200, n_features=5, noise=10.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reg = LinearRegression()
    
    va_ensemble = VennAbersCV(estimator=reg, inductive=False, n_splits=3, setting='regression', cv_ensemble=True, shuffle=True, random_state=42, m_parameter=1)
    va_ensemble.fit(X_train, y_train)
    mid_ensemble, range_ensemble = va_ensemble.predict_interval(X_test)
    
    va_single = VennAbersCV(estimator=reg, inductive=False, n_splits=3, setting='regression', cv_ensemble=False, shuffle=True, random_state=42, m_parameter=1)
    va_single.fit(X_train, y_train)
    mid_single, range_single = va_single.predict_interval(X_test)
    
    # Both should run without error and return correct shapes
    assert len(mid_ensemble) == 3
    assert len(mid_single) == 3
    # Check that prediction outputs are actually different due to cv_ensemble differences
    # (since ensemble uses fold estimators, single uses full estimator)
    assert not np.allclose(mid_ensemble[0], mid_single[0])


def test_venn_abers_cv_epsilon_regression():
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from venn_abers import VennAbersRegressor
    
    X, y = make_regression(n_samples=200, n_features=5, noise=10.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reg = LinearRegression()
    
    # Fit with default m=1
    va_m1 = VennAbersRegressor(estimator=reg, inductive=False, n_splits=3, random_state=42)
    va_m1.fit(X_train, y_train, m=1)
    mid_m1, range_m1 = va_m1.predict(X_test)
    
    # Fit with epsilon=0.5
    va_eps = VennAbersRegressor(estimator=reg, inductive=False, n_splits=3, random_state=42)
    va_eps.fit(X_train, y_train, epsilon=0.5)
    mid_eps, range_eps = va_eps.predict(X_test)
    
    # The output intervals should NOT be identical because epsilon=0.5 leads to m=13 in cross path folds (k=53/54)
    # whereas default/m=1 uses m=1.
    assert not np.allclose(range_m1, range_eps)
    
    # Let's also verify floor conversion: for k=53, epsilon=0.5, floor(0.5 * 54 / 2) = 13.
    # If we used round, for some cases it might round up. Let's verify our specific math.
    # We can inspect the va_calibrator_ internal states if needed, but the range difference is the main thing.
    assert va_eps.va_calibrator_.epsilon == 0.5


def test_venn_abers_prediction_above_range():
    from venn_abers import calc_p0p1, calc_probs
    import numpy as np
    
    # Test regression setting
    p_cal = np.array([1.0, 2.0, 3.0])
    y_cal = np.array([10.0, 20.0, 15.0])
    p0, p1, c = calc_p0p1(p_cal, y_cal, setting='regression')
    
    # The last element of p1 must be the max label (y*), which is 20.0 (not the GCM slope 18.33333...)
    assert p1[-1, 1] == 20.0
    
    # Check that calc_probs yields correct upper bound when out > c[-1]
    p_prime, p0_p1 = calc_probs(p0, p1, c, np.array([4.0]), setting='regression')
    assert p0_p1[0, 1] == 20.0
    
    # Test classification setting
    p_cal_cls = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    y_cal_cls = np.array([0, 1, 0])
    p0_cls, p1_cls, c_cls = calc_p0p1(p_cal_cls, y_cal_cls, setting='classification')
    
    # The last element of p1 must be 1.0 (not the GCM slope 0.6666...)
    assert p1_cls[-1, 1] == 1.0
    
    # Check that calc_probs yields 1.0 when out > c[-1]
    p_prime_cls, p0_p1_cls = calc_probs(p0_cls, p1_cls, c_cls, np.array([[0.6, 0.4]]), setting='classification')
    assert p0_p1_cls[0, 1] == 1.0
