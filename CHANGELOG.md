# Changelog

## [1.5.4] - 2026-08-12

### Fixed
- **`VennAbersCV` Estimator Cloning**: Cloned base estimators using `sklearn.base.clone` at each cross-validation fold to prevent state mutation across folds.
- **Cross-Validation Fold Test Predictions**: Fixed `predict_interval` in `VennAbersCV` to evaluate test samples on each fold's respective fitted estimator rather than evaluating only against the final fold's estimator.
- **Ignored Epsilon in Cross-Validation Path**: Enabled `epsilon` to $m$ conversion for `VennAbersRegressor` when `inductive=False`, and updated the calculation to use `np.floor` instead of `np.round` ($m = \lfloor \varepsilon(k+1)/2 \rfloor$) in line with theoretical formulations.
- **Upper Bound $f^*$ Out-of-Range Regression Bug**: Fixed `calc_p0p1` so test predictions exceeding all calibration predictions correctly return the maximum calibration label $y^*$ rather than an invalid GCM slope.
- **Legacy Typo**: Fixed legacy references importing `VennAberRegressor` (singular) instead of `VennAbersRegressor` in example notebooks.

### Added
- **Export Core Functions in Package Namespace**: Exposed `calc_p0p1` and `calc_probs` in top-level `venn_abers` namespace.
- **Comprehensive Unit Tests**: Added regression tests in `tests/test_venn_abers.py` verifying cloning isolation across CV folds, epsilon behavior, and boundary regression predictions.

### Changed
- Refreshed and re-executed all example notebooks in `examples/` with updated outputs and plots.

## [1.5.3] - 2026-05-08


### Added
- **GitHub Actions CI**: Automated test suite that runs on every push and pull request.
- **Integrity Tests**: New test suite (`tests/test_integrity.py`) ensuring numeric consistency of predictions against baseline implementation.
- **Type Hints**: Python type hints added to `VennAbers` class methods for improved developer experience.

### Fixed
- **Docstring Refactoring**: Moved misplaced docstrings to follow standard Python conventions (PR #36).
- **Test Typos**: Fixed `VennAberRegressor` naming typos in the existing test suite.
- **Cleanup**: Removed unused notebook placeholders.

### Changed
- Standardized project structure for better CI/CD integration.
