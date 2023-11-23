import numpy as np

class Node():
  """
  A class representing a node of a decision tree model.
  """
  def __init__(self, gini = 0, entropy = 0, num_samples = 0, threshold = None,
               num_samples_per_class = None, predicted_class = None,
               feature_index = None, left = None, right = None):
    self.gini = gini
    self.entropy = entropy
    self.num_samples = num_samples
    self.num_samples_per_class = num_samples_per_class
    self.predicted_class = predicted_class
    self.feature_index = feature_index 
    self.threshold = threshold
    self.left = left
    self.right = right


class DecisionTreeC():
  """
  A class representing a decision tree model.
  """

  def __init__(self, criterion = 'gini', splitter = 'best', max_depth = None,
               min_samples_split = 2, min_samples_leaf = 1, max_features = None):
    self.criterion = criterion
    self.splitter = splitter
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.min_samples_leaf = min_samples_leaf
    self.max_features = max_features
    self.tree_ = None

  def fit(self,X,y):
    self.classes_ = list(set(y))
    self.n_classes_ = len(set(y))
    self.n_features_ = X.shape[1]
    self.tree_ = self._grow_tree(X,y)
  
  def _grow_tree(self,X,y,depth=0):
    num_samples_per_class = [np.sum(y==cls) for cls in self.classes_]
    predicted_class = np.argmax(num_samples_per_class)
    
    # Base node
    node = Node (
      gini = self._gini(y),
      entropy = self._entropy(y),
      num_samples = len(y),
      num_samples_per_class = num_samples_per_class,
      predicted_class = predicted_class
    )

    if depth == self.max_depth: # Check max_depth
      return node
    if node.num_samples < self.min_samples_split: # Check min_samples_split
      return node
    if np.max(num_samples_per_class) == node.num_samples: # Check all samples are from the same class
      return node
    
    # If the previous criteria is not met we search for the best split
    best_feature, best_threshold = self._best_split(X,y)

    if best_feature is None:
      return node
    
    # Create child nodes
    left_indices = X[:,best_feature] < best_threshold
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[~left_indices], y[~left_indices]

    if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf: #Check if either side is too small. If so, reutrn a leaf node and don't split
      return node

    # Recursively create new nodes
    node.feature_index = best_feature
    node.threshold = best_threshold
    node.left = self._grow_tree(X_left,y_left,depth=depth+1)
    node.right = self._grow_tree(X_right,y_right,depth=depth+1)

    return node

  def _best_split(self,X,y):
    if self.splitter == 'best':
      # If self.max_features is None, consider all features. Otherwise randomly pick the corresponding n_features_to_pick
      features_to_consider = range(self.n_features_)
      if self.max_features is not None: #Consider all features
        if self.max_features == 'sqrt':
          n_features_to_pick = int(np.sqrt(self.n_features_))
        elif self.max_features == 'log2':
          n_features_to_pick = int(np.log2(self.n_features_))
        else: # Assuming it is an integer
          n_features_to_pick = self.max_features
        
        features_to_consider = np.random.choice(features_to_consider,n_features_to_pick)
      
      best_impurity = float('inf')
      best_feature_idx = None
      best_threshold = None
      
      for idx in features_to_consider:
        possible_thresholds = set(X[:,idx])
        for threshold in possible_thresholds:
          # Split data based on current feature and threshold
          left_mask = X[:,idx] < threshold
          y_left, y_right = y[left_mask], y[~left_mask]

          # Calculate the weighted impurity for the split
          left_impurity = self._calculate_impurity(y_left)
          right_impurity = self._calculate_impurity(y_right)
          weighted_impurity = (len(y_left)/len(y))*left_impurity + (len(y_right)/len(y))*right_impurity

          # Update values if the weighted entropy is better than best impurity
          if weighted_impurity < best_impurity:
            best_impurity = weighted_impurity
            best_feature_idx = idx
            best_threshold = threshold
      
      return best_feature_idx, best_threshold
    
    elif self.splitter == 'random':
      # Randomly select a feature and a threshold
      feature_idx = np.random.randint(self.n_features_)
      threshold = np.random.choice(set(X[:,feature_idx]))

      return feature_idx, threshold
    
    else:
      raise ValueError("Please select a valid splitter option: 'best' or 'random'")
    
  def _calculate_impurity(self,y):
    if self.criterion == 'gini':
      return self._gini(y)
    elif self.criterion == 'entropy':
      return self._entropy(y)
    else:
      raise ValueError(f"Unknown criterion {self.criterion}")
  
  def _gini(self,y):
    n_samples_ = len(y)
    num_samples_per_class = [np.sum(y == cls) for cls in self.classes_]
    gini = 1 - np.sum(np.array(num_samples_per_class)**2) / n_samples_**2

    return gini
  
  def _entropy(self,y):
    n_samples_ = len(y)
    num_samples_per_class = np.array([np.sum(y == cls) for cls in self.classes_])
    num_samples_per_class_ratio = num_samples_per_class / n_samples_
    ratios_non_zero = np.clip(num_samples_per_class_ratio, a_min=1e-10, a_max=None) # Avoid taking log2(0) by adding a small value
    entropy = - np.dot(ratios_non_zero,np.log2(ratios_non_zero))

    return entropy
  
  def predict(self,X):
    return np.array([self._predict_sample(sample,self.tree_) for sample in X])
  
  def _predict_sample(self,sample,node):
    if node.left is None and node.right is None: #Leaf node
      return node.predicted_class
    if sample[node.feature_index] < node.threshold:
      return self._predict_sample(sample,node.left)
    else:
      return self._predict_sample(sample,node.right)





        
        