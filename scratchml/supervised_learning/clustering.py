import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import mode

class KNNClassifier():
    """
    A class implementing KNN (K-Nearest-Neighbors) algorithm for classification.

    Attributes:
        n_neighbors (int): Number of neighbors to use for classification
        distance (str): The distance metric to use. Currently only the euclidean distance is supported."  
    
    Methods:
        predict(X_pred, X_train, y_train): Predicts the class labels for the target data.

    """
    def __init__(self, n_neighbors: int = 5, distance: str = 'euclidean'):
        self.n_neighbors = n_neighbors
        self.distance = distance
    
    def _calculate_distance(self, instance, X_train):
        if self.distance == 'euclidean':
            distances = np.sqrt(np.sum((X_train - instance) ** 2, axis = 1))
        else:
            raise ValueError("Invalid distance method. Please choose one of the following: euclidean")
        return distances
            
    
    def _vote_class(self, idxs, y_train):
        nearest_classes = y_train[idxs]
        pred_class = mode(nearest_classes)
        return pred_class.mode

    def predict(self,X_pred, X_train, y_train):
        y_pred = np.empty(X_pred.shape[0])
        for idx, instance in enumerate(X_pred):
            distances = self._calculate_distance(instance, X_train)
            nearest_neighbor_idxs = np.argsort(distances)[:self.n_neighbors]
            y_pred[idx] = self._vote_class(nearest_neighbor_idxs,y_train)
        
        return y_pred
    
    def visualize_classification(self, X_train, y_train, X_test, y_test):
        """
        Visualize the classification process of the KNN algorithm. Apply pca with 2 components and then use the function.
        Otherwise it will use the first 2 features of the dataset for the visualization.
        """
        # Plot the training data
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='Training data', alpha=0.5)

        # Initialize markers for correct and incorrect predictions
        correct_marker = '^'
        incorrect_marker = 'x'

        # Predict and plot each test data point
        for idx, instance in enumerate(X_test):
            prediction = self.predict(np.array([instance]), X_train, y_train)
            actual = y_test[idx]
            marker = correct_marker if prediction == actual else incorrect_marker
            plt.scatter(instance[0], instance[1], c='red' if prediction else 'blue', marker=marker)

        # Create custom legend
        correct_legend = mlines.Line2D([], [], color='green', marker=correct_marker, linestyle='None', markersize=10, label='Correct Prediction')
        incorrect_legend = mlines.Line2D([], [], color='red', marker=incorrect_marker, linestyle='None', markersize=10, label='Incorrect Prediction')
        plt.legend(handles=[correct_legend, incorrect_legend])

        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        plt.title('KNN Classification Visualization')
        plt.show()
