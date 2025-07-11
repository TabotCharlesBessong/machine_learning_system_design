
import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
  return np.sqrt(np.sum((x1-x2)**2))

class KNN:
  def __init__(self,k=3):
    self.k = k
    
  def fit(self,X,y):
    self.X_train = X
    self.y_train = y
    
  def predict(self,test):
    return [self.predict_label(x) for x in test]
  
  def predict_label(self,test):
    # compute the distances
    distances = [euclidean_distance(test,x) for x in self.X_train]
    # sort by distance and return indices of the first k neighbors
    k_indices = np.argsort(distances)[:self.k]
    # extract the labels of the k nearest neighbor training samples
    k_nearest_labels = [self.y_train[i] for i in k_indices]
    # return the most common class
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]