import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from venn_abers import VennAbersCalibrator, VennAbersRegressor

def test_prediction_integrity_binary():
    """Ensure classification predictions match known baseline for a fixed seed."""
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = GaussianNB().fit(X_train, y_train)
    va = VennAbersCalibrator(estimator=clf, inductive=True, cal_size=0.2, random_state=42)
    va.fit(X_train, y_train)
    p_prime = va.predict_proba(X_test)
    
    # Expected values from main branch baseline
    expected_sample = np.array([[0.8, 0.2], [0.8, 0.2]])
    np.testing.assert_allclose(p_prime[:2], expected_sample, atol=1e-10)

def test_prediction_integrity_regression():
    """Ensure regression predictions match known baseline for a fixed seed."""
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression()
    va_reg = VennAbersRegressor(estimator=reg, inductive=True, cal_size=0.2, random_state=42)
    va_reg.fit(X_train, y_train)
    mid, interval = va_reg.predict(X_test)
    
    # Expected values from main branch baseline
    expected_mid_sample = np.array([-26.76237444208006, -1.751374389088526])
    expected_interval_sample = np.array([
        [-75.26333237302251, -26.76237444208006],
        [-1.751374389088526, 48.268529815049365]
    ])
    
    np.testing.assert_allclose(mid[:2], expected_mid_sample, atol=1e-10)
    np.testing.assert_allclose(interval[:2], expected_interval_sample, atol=1e-10)
