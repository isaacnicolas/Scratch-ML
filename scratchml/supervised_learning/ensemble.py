import numpy as np
import scipy

from scratchml.supervised_learning.tree import DecisionTreeC, DecisionTreeR

from scratchml.utils.base import clone_estimator

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
    
class AdaBoostC():
    """
    A class implementing an AdaBoost classifier.

    Parameters:

    """
    def __init__(self, estimator: object = None, n_estimators: int = 50,
                 learning_rate: float = 1.0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = None
        self.weights = None
        self.pred_weights = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        # Set initial weights
        weights = np.array([1/len(X) for _ in range(len(X))])
        self.pred_weights = []
        # Initialize estimators 
        self.estimators = [clone_estimator(estimator = self.estimator) for _ in range(self.n_estimators)]
        for estimator in self.estimators:
            X_sampled, y_sampled = self._sample_data(X, y, weights)
            estimator.fit(X_sampled, y_sampled)
            y_pred = estimator.predict(X_sampled)
            weights, predictor_weight = self._update_weights(y_pred, y_sampled, weights)
            self.pred_weights.append(predictor_weight)

    def predict(self, X):
        # Initialize a matrix to store the predictions
        class_predictions = np.zeros((X.shape[0], self.n_classes_))
        # For each estimator, predict values. For each predicted sample sum weight to the corresponding class
        for estimator, weight in zip(self.estimators,self.pred_weights):
            predictions = estimator.predict(X)
            for i in range (X.shape[0]):
                class_predictions[i, predictions[i]] += weight
        # Final predictions
        final_predictions = np.argmax(class_predictions, axis = 1)

        return final_predictions

    def _sample_data(self, X, y, weights):
        # Normalize weights so that they sum 1
        norm_weights = weights / np.sum(weights)

        # Get sampling indices according to weights
        sample_indices = np.random.choice(np.arange(len(X)), size = len(X), p = norm_weights)

        # Sample data
        X_sampled = X[sample_indices]
        y_sampled = y[sample_indices]
        
        return X_sampled, y_sampled
    
    def _update_weights(self, y_pred, y, weights):
        # Calculate weighted error rate
        indicator = (y_pred != y).astype(int)
        weighted_error_rate = np.sum(weights * indicator) / np.sum(weights)
        # Avoid division by zero
        weighted_error_rate = np.clip(weighted_error_rate, 1e-10, 1 - 1e-10)
        # Calculate predictor's weight
        predictor_weight = self.learning_rate * np.log((1 - weighted_error_rate) / weighted_error_rate)
        # Update weights
        weights *= np.exp(predictor_weight * indicator)
        weights /= np.sum(weights)
 
        return weights, predictor_weight

class AdaBoostR():
    """
    A class implementing an AdaBoost Regressor.

    Parameters:
        estimator (object): Estimator object to use sequentially. Should include fit and predict methods.
        n_estimators (int): Number of estimators to train sequentially.
        learning_rate (float): Learning rate of predictor weight.

    """
    def __init__(self, estimator: object = None, n_estimators: int = 50,
                 learning_rate: float = 1.0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.weights = []
        self.pred_weights = []

    def fit(self, X, y):
        # Set initial weights
        weights = np.array([1/len(X) for _ in range(len(X))])
        # Initialize estimators 
        self.estimators = [clone_estimator(estimator = self.estimator) for _ in range(self.n_estimators)]
        
        for estimator in self.estimators:
            X_sampled, y_sampled = self._sample_data(X, y, weights)
            
            estimator.fit(X_sampled, y_sampled)
            y_pred = estimator.predict(X_sampled)
            
            weights, predictor_weight = self._update_weights(y_pred, y_sampled, weights)
            self.pred_weights.append(predictor_weight)

    def predict(self, X):
        predictions = np.zeros(len(X))

        for estimator, weight in zip(self.estimators, self.pred_weights):
            predictions += estimator.predict(X) * weight
        return predictions / np.sum(self.pred_weights)

    def _sample_data(self, X, y, weights):
        # Normalize weights so that they sum 1
        norm_weights = weights / np.sum(weights)

        # Get sampling indices according to weights
        sample_indices = np.random.choice(np.arange(len(X)), size = len(X), p = norm_weights)

        # Sample data
        X_sampled = X[sample_indices]
        y_sampled = y[sample_indices]
        
        return X_sampled, y_sampled
    
    def _update_weights(self, y_pred, y, weights):
        """
        Reference: Improving Regressors Using Boosting Techniques (1997)
        """
        # Calculate absolute errors
        abs_errors = np.abs(y_pred - y)

        # Calculate supremum value
        sup = np.max(abs_errors)

        # Calculate the losses (exponential)
        losses = np.array(1 - (np.exp((-abs_errors) / sup)))

        # Avoid division by zero and extreme weights
        losses = np.clip(losses, 1e-10, 1 - 1e-10)

        # Average loss
        loss = np.average(losses)

        # Calculate predictor's weight
        predictor_weight = self.learning_rate * (loss / (1 - loss))

        # Update weights
        weights *= predictor_weight**(1 - losses)
        weights /= np.sum(weights)

        return weights, predictor_weight