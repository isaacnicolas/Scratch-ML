import numpy as np
import scipy

class KNNClassifier():
    """
    A class implementing KNN (K-Nearest-Neighbors) algorithm.

    Attributes:

    """
    def init(self, n_neighbors: int = 3, distance: str = 'euclidean',
             ):
        self.n_neighbors = n_neighbors
        self.distance = distance
