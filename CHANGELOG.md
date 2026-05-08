# Changelog

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
