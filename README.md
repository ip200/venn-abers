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


### Examples
Further examples can be found in the github repository https://github.com/ip200/venn-abers in the `examples` folder:

- [simple_classification.ipynb](https://github.com/ip200/venn-abers/blob/main/examples/simple_classification.ipynb) for a simple example in the binary classification setting.
- [multiclass_classification.ipynb](https://github.com/ip200/venn-abers/blob/main/examples/multiclass_classification.ipynb) for the multiclass setting.
- [comparison_with_existing.ipynb](https://github.com/ip200/venn-abers/blob/main/examples/comparison_with_existing.ipynb) for the comparison with a previous github implementation


### Citation
If you find this library useful please consider citing:

- Vovk, Vladimir, Ivan Petej and Valentina Fedorova. "Large-scale probabilistic predictors with and without guarantees of validity." Advances in Neural Information Processing Systems 28 (2015) (arxiv version https://arxiv.org/pdf/1511.00213.pdf)
- Vovk, Vladimir, Ivan Petej. "Venn-Abers predictors". Proceedings of the Thirtieth Conference on Uncertainty in Artificial Intelligence (2014) (arxiv version https://arxiv.org/abs/1211.0025)
- Manokhin, Valery. "Multi-class probabilistic classification using inductive and cross Venn–Abers predictors." Conformal and Probabilistic Prediction and Applications, PMLR, 2017.