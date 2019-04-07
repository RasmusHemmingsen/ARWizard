import numpy as np
import pandas as pd
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import add_dummy_feature
from sklearn.metrics import accuracy_score
import pickle

#Load the provided data from .mat files
def loadData():
    train_data = pd.read_csv(r"AR Wizard\Assets\TrainingData\HandGestureData.csv")
    train_label = np.array([0]*120 + [1]*131 + [2]*131 + [3]*130 + [4]*127)

    return train_data, train_label

class MSE_Perceptron:
    def __init__(self):
        self.label_offset = 0
        self.W = np.zeros(1)

    def fit(self, train_data, train_lbls, epsilon):
        # Create set of training classes
        classes = np.unique(train_lbls)

        # Convert samples to float for faster numpy processing
        train_data = train_data.astype(float)

        # Augment data with bias to simplify linear discriminant function
        X = add_dummy_feature(train_data).transpose()
        n_features, n_samples = X.shape

        # Calculate regularized pseudo-inverse of X
        X_pinv = np.dot(np.linalg.inv(np.dot(X, X.transpose()) + epsilon * np.identity(n_features)), X)

        # Initialize target matrix B for OVR (One vs Rest) binary classification
        B = np.where(train_lbls[np.newaxis, :] == classes[:, np.newaxis], 1, -1)

        # Calculate optimized weight vectors
        self.W = np.dot(B, X_pinv.transpose())
        return self

    def predict(self, test_data):
        return perceptron_classify(self.W, test_data)

def perceptron_classify(W, test_data):
    # Convert samples to float for faster numpy processing
    test_data = test_data.astype(float)
    
    # Augment data with bias to simplify linear discriminant function
    X = add_dummy_feature(test_data).transpose()

    decision = np.dot(W,X)
    return np.argmax(decision,axis=0)

train_data, train_label = loadData()

mse = MSE_Perceptron()

mse.fit(train_data, train_label, 100)

filename = 'model.sav'
pickle.dump(mse, open(filename, 'wb'))