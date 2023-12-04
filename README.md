# ScratchML
## About
This repository contains implementations of some of the most popular ML algorithms from scratch. The purpose is to deepen the understanding of the concepts underlying this algorithms.

## Contents
- [About](#about)
- [Supported models](#supported-models)
- [Examples](#examples)
- [Installation](#installation)

## Supported models:
The * means that the model has implementations both for regression and classification.
### Supervised Learning:
- [Linear Regression](scratchml/supervised_learning/linear.py)
- [Logistic Regression](scratchml/supervised_learning/linear.py)
- [Decision Tree*](scratchml/supervised_learning/tree.py)
- [Random Forest*](scratchml/supervised_learning/ensemble.py)
- [Adaboost*](scratchml/supervised_learning/ensemble.py)
- [XGBoost (eXtreme Gradient Boosting)]()
- [Gradient Boosting]()
- [KNN (K Nearest Neighbors)]()
### Unsupervised Learning:
- [DBSCAN]()
- [K-Means]()
- [PCA (Principal component analysis)]()

## Examples
See the **examples** folder for notebooks that use the implementations.
- Linear Regression [Notebook](examples/LinearRegression.ipynb)
- Logistic Regression [Notebook](examples/LogisticRegression.ipynb)
- Decision Tree and Random Forest [Notebook](examples/DecisionTree.ipynb)
- AdaBoost [Notebook](examples/AdaBoost.ipynb)

## Installation
```shell
git clone https://github.com/isaacnicolas/scratchml.git
cd scratchml
pip install -e .
```