[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Static Badge](https://img.shields.io/badge/Probabilistic_Calibration-blue)
![GitHub Repo stars](https://img.shields.io/github/stars/ip200/venn-abers)




# Venn-ABERS calibration
This library contains the Python implementation of Venn-ABERS calibration for binary and multiclass classification problems.

### Installation
```commandline
pip install venn-abers
```
The method can be applied on top of an underlying scikit-learn algorithm.
### Example Usage
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from venn_abers import VennAbersCalibrator

X, y = make_classification(n_samples=1000, n_classes=3, n_informative=10)
X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = GaussianNB()

# Define Venn-ABERS calibrator
va = VennAbersCalibrator(estimator=clf, inductive=True, cal_size=0.2, random_state=101)

# Fit on the training set
va.fit(X_train, y_train)

# Generate probabilities and class predictions on the test set
p_prime = va.predict_proba(X_test)
y_pred = va.predict(X_test)
```

### Scikit-Learn Compatibility
The `venn-abers` library is fully compatible natively with the `scikit-learn` ecosystem! It seamlessly inherits from `BaseEstimator` allowing it to be dropped directly into workflows like `Pipeline`, `GridSearchCV`, and `cross_val_score`. 

Furthermore, this compatibility is **100% backwards-compatible** with all previous implementations of the library, so no existing code will break.

You can also pass `**kwargs` directly into the `.fit()` method to route parameters dynamically to the underlying estimator (e.g. passing `eval_set` to XGBoost):

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from venn_abers import VennAbersCalibrator

# Passing kwargs through fit explicitly
va = VennAbersCalibrator(estimator=XGBClassifier(), inductive=True, cal_size=0.2)
va.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)

# Scikit-learn Pipeline Integration
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('va', VennAbersCalibrator(estimator=GaussianNB(), inductive=True, cal_size=0.2))
])
pipeline.fit(X_train, y_train)
p_prime_pipeline = pipeline.predict_proba(X_test)
```

### Examples
Further examples can be found in the github repository https://github.com/ip200/venn-abers in the `examples` folder:

- [simple_classification.ipynb](https://github.com/ip200/venn-abers/blob/main/examples/simple_classification.ipynb) for a simple example in the binary classification setting.
- [multiclass_classification.ipynb](https://github.com/ip200/venn-abers/blob/main/examples/multiclass_classification.ipynb) for the multiclass setting.
- [comparison_with_existing.ipynb](https://github.com/ip200/venn-abers/blob/main/examples/comparison_with_existing.ipynb) for the comparison with a previous github implementation.
- [ivar_example.ipynb](https://github.com/ip200/venn-abers/blob/main/examples/ivar_example.ipynb) for an example of Inductive Venn-ABERS for regression.
- [scikit_learn_integration.ipynb](https://github.com/ip200/venn-abers/blob/main/examples/scikit_learn_integration.ipynb) for an example of the scikit-learn native integration.


## Academic Usage

The `venn-abers` library has been used or referenced in several academic works related to probability calibration, uncertainty estimation, and conformal prediction.

### Papers Referencing This Repository

- **Kazantsev, D. (2025)**  
  [*Adaptive Set‑Mass Calibration with Conformal Prediction.*](https://arxiv.org/abs/2505.15437)  
  arXiv:2505.15437

- **Rabenseifner, M. (2025)**  
  [*Calibration Strategies for Robust Causal Estimation: Theoretical and Empirical Insights on Propensity Score Based Estimators.*](https://arxiv.org/abs/2503.17290)  
  arXiv:2503.17290

- **Löfström et al. (2024)**  
  [*Calibrated Explanations: With Uncertainty Information and Counterfactuals.*](https://doi.org/10.1016/j.eswa.2024.123154)  
  Expert Systems with Applications.

---

### Research Using Venn‑Abers Calibration

These papers apply Venn‑Abers, IVAP, or CVAP calibration methods.

- **Manokhin and Grønhaug (2025)**  
  [*Classifier Calibration at Scale: An Empirical Study of Model-Agnostic Post-Hoc Methods.*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6094047)

- **Boger et al. (2025)**  
  [*Functional protein mining with conformal guarantees.*](https://www.nature.com/articles/s41467-024-55676-y)  

- **van der Laan & Alaa (2025)**  
  [*Generalized Venn and Venn‑Abers Calibration with Applications in Conformal Prediction.*](https://arxiv.org/abs/2502.05676)

- **Johansson, Löfström & Sönströd (2023)**  
  [*Well‑Calibrated Probabilistic Predictive Maintenance using Venn‑Abers.*](https://arxiv.org/abs/2306.06642)

- **Pereira et al. (2020)**  
  [*Targeting the Uncertainty of Predictions at Patient Level.*](https://doi.org/10.1016/j.jbi.2019.103350)  
  Journal of Biomedical Informatics.

For a more comprehensive list please see [*Google Scholar*](https://scholar.google.com/scholar?cites=7349763935201730660&as_sdt=2005&sciodt=0,5&hl=en)

---

### Related Libraries

- **MAPIE – Model Agnostic Prediction Interval Estimator**  
  MAPIE includes calibration utilities referencing Venn‑Abers predictors.  
  https://github.com/scikit-learn-contrib/MAPIE

---

## Foundational Research

- Vovk & Petej (2014) — [*Venn‑Abers Predictors.*](https://arxiv.org/abs/1211.0025)
- Vovk, Petej & Fedorova (2015) — [*Large‑scale probabilistic predictors with validity guarantees.*](https://arxiv.org/abs/1511.00213)
- Manokhin (2017) — [*Multi‑class probabilistic classification using inductive and cross Venn‑Abers predictors.*](https://proceedings.mlr.press/v60/manokhin17a.html)

---

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{petej_venn_abers,
  author = {Petej, Ivan},
  title = {venn-abers: Venn-Abers calibration for probabilistic classifiers},
  year = {2024},
  url = {https://github.com/ip200/venn-abers}
}
```

You may also cite the foundational paper:

```bibtex
@inproceedings{vovk2014venn,
  title={Venn-Abers predictors},
  author={Vovk, Vladimir and Petej, Ivan},
  booktitle={Proceedings of the Thirtieth Conference on Uncertainty in Artificial Intelligence},
  pages={829--838},
  year={2014}
}
```

---

## Contributing

Contributions and pull requests are welcome.  
If you are aware of additional research using this library, please open a PR to add it to the **Academic Usage** section.

---

## License

MIT License
