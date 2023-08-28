# venn-abers
This libraty contains the Python implementation of Venn-ABERS calibration for binary and multiclass classification problems.

### Usage
```commandline

X, y = make_classification(n_samples=1000, n_classes=3, n_informative=10)
X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = GaussianNB()

va = VennAbersCalibrator(estimator=clf, inductive=True, cal_size=0.2, random_state=101)
va.fit(X_train, y_train)

p_prime = va.predict_proba(X_test)
y_pred = va.predict(X_test)
```


### Examples
You can find complete code examples in the examples folder. Refer to:

- examples/simple_classification.ipynb for a simple example in the binary classification setting.
- examples/multiclass_classification.ipynb for the multiclass (more than 2 classes) setting.
- examples/comparison_with_existing.py for the comparison with an existing github implementation

### Citation
If you find this library useful please consider citing:

- Vovk, Vladimir, Ivan Petej, and Valentina Fedorova. "Large-scale probabilistic predictors with and without guarantees of validity."Advances in Neural Information Processing Systems 28 (2015)(arxiv version https://arxiv.org/pdf/1511.00213.pdf)
- Vovk, Vladimir, Ivan Petej "Venn-Abers predictors". Proceedings of the Thirtieth Conference on Uncertainty in Artificial Intelligence (2014) Pages 829–838 (arxiv version https://arxiv.org/abs/1211.0025)
- Manokhin, Valery. "Multi-class probabilistic classification using inductive and cross Venn–Abers predictors." In Conformal and Probabilistic  Prediction and Applications, pp. 228-240. PMLR, 2017.