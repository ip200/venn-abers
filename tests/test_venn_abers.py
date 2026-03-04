import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from venn_abers import VennAbers, VennAbersCV, VennAbersMultiClass, VennAbersCalibrator, VennAberRegressor

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
    va_reg = VennAberRegressor(estimator=reg, inductive=True, cal_size=0.2, random_state=42)
    va_reg.fit(X_train, y_train)
    
    mid, interval = va_reg.predict(X_test)
    
    assert mid.shape == (len(X_test),)
    assert interval.shape == (len(X_test), 2)
    assert np.all(interval[:, 0] <= interval[:, 1])

def test_venn_abers_regressor_cvap(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    
    reg = LinearRegression()
    # Adding epsilon to not have edge cases fail
    va_reg = VennAberRegressor(estimator=reg, inductive=False, n_splits=3, random_state=42)
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
