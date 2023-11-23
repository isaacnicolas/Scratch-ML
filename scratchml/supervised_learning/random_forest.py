import numpy as np
import scipy
from scratchml.supervised_learning.tree import DecisionTreeC, DecisionTreeR

class RandomForestC():
    """
    A class implementing a Random Forest Classifier.

    Attributes:
        n_estimators (int): The number of trees in the forest.
        criterion (str): The function to measure the quality of a split.
        splitter (str): The strategy used to split a node.
        max_depth (int): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        max_features: The number of features to consider when looking for the best split.
    """
    def __init__(self, n_estimators: int = None, criterion: str = 'gini', splitter: str = 'best',
                 max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: int = None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = [] 
    
    def fit(self,X,y):
        self.trees = [DecisionTreeC(criterion = self.criterion,
                                    splitter = self.splitter,
                                    max_depth = self.max_depth,
                                    min_samples_split = self.min_samples_split,
                                    min_samples_leaf = self.min_samples_leaf,
                                    max_features = self.max_features)
                    for _ in range(self.n_estimators)]
        
        for tree in self.trees:
            # Bootstraping: sample data with replacement
            bootstrap_indices = np.random.randint(0,len(X),len(X))
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree.fit(X_bootstrap,y_bootstrap)
    
    def predict(self,X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Aggregate predictions: Take the mode of the predictions
        tree_preds_mode = scipy.stats.mode(tree_preds, axis=0)[0]
        
        return tree_preds_mode.flatten()
    
class RandomForestR():
    """
    A class implementing a Random Forest Regressor.

    Attributes:
        n_estimators (int): The number of trees in the forest.
        criterion (str): The function to measure the quality of a split.
        splitter(str): The strategy used to split a node.
        max_depth (int): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        max_features: The number of features to consider when looking for the best split.
    """
    def __init__(self, n_estimators: int = None, criterion: str = 'mse', splitter: str = 'best',
                 max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: int = None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = [] 
    
    def fit(self,X,y):
        self.trees = [DecisionTreeR(criterion = self.criterion,
                                    splitter = self.splitter,
                                    max_depth = self.max_depth,
                                    min_samples_split = self.min_samples_split,
                                    min_samples_leaf = self.min_samples_leaf,
                                    max_features = self.max_features)
                    for _ in range(self.n_estimators)]
        
        for tree in self.trees:
            # Bootstraping: sample data with replacement
            bootstrap_indices = np.random.randint(0,len(X),len(X))
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree.fit(X_bootstrap,y_bootstrap)
    
    def predict(self,X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Aggregate predictions: Take the mode of the predictions
        tree_preds_mean = np.mean(tree_preds, axis=0)
        
        return tree_preds_mean