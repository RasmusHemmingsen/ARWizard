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
    train_data = pd.read_csv(r"C:\Users\Nicklas\Dropbox\Computer Teknologi\CT 3. Semester\HandGestureDataGestureTogether.csv")
    train_label = np.array([0]*103 + [1]*98 + [2]*106)

    test_data = pd.read_csv(r"C:\Users\Nicklas\Dropbox\Computer Teknologi\CT 3. Semester\HandGestureTestSetAnders.csv")
    test_label = np.array([0]*9 + [1]*9 + [2]*10)

    return train_data, train_label, test_data, test_label


# Nearest Centroid Algorithm
class NC:
    def __init__(self):
        self.clf = NearestCentroid()
        self.centroids = []

    # Calculates the mean of each class in the training data. 
    def fit(self, train_data, train_lbls):
        self.clf.fit(train_data, train_lbls)
        self.centroids = self.clf.centroids_
        return self

    def predict(self, test_data, test_lbls):
        classification = self.clf.predict(test_data)
        try:
            score = accuracy_score(test_lbls, classification)
        except ValueError:
            score = None

        return classification, score

# Nearest Sub Centroid
class NSC:
    def __init__(self, subclass_count):
        self.kmeans = KMeans(n_clusters=subclass_count)
        self.subclass_centers = []
        self.label_offset = 0
        self.classes = []
        self.subclass_count = subclass_count

    def fit(self, train_data, train_lbls):
        # Create set of training classes
        self.classes = np.unique(train_lbls)
        class_count = len(self.classes)
        n_samples, n_features = train_data.shape

        # Iterate classes and apply K-means to find subclasses of each class
        grouped_train_data = [None] * class_count
        self.subclass_centers = np.zeros((class_count, self.subclass_count, n_features))
        self.label_offset = self.classes[0]   # Account for classifications which doesn't start at 0
        for label in self.classes:
            index = label - self.label_offset

            # Group training samples into lists for each class
            grouped_train_data[index] = [x for i, x in enumerate(train_data) if train_lbls[i]==label]

            # Apply K-means clustering algorithm to find subclasses
            self.kmeans.fit(grouped_train_data[index])
            self.subclass_centers[index] = self.kmeans.cluster_centers_
        return self

    def predict(self, test_data, test_lbls):
        class_count = len(self.classes)
        n_samples, n_features = test_data.shape
        distances = np.zeros((class_count, n_samples))

        # Iterate classes and calculate distances to each subclass centroid
        for k in range(class_count):
            class_distances = np.sqrt(((test_data - (self.subclass_centers[k,:,:])[:,np.newaxis,:]) ** 2).sum(axis=2))
            distances[k] = np.min(class_distances, axis=0)

        # Classify samples to class with closes subclass centroid
        classification = np.argmin(distances,axis=0) + self.label_offset

        # Determine classification errors by comparing classification with known labels
        try:
            score = accuracy_score(test_lbls, classification)
        except ValueError:
            score = None

        return np.asarray(classification), score

#Nearest Neighbor
class NN:
    def __init__(self):
        self.clf = KNeighborsClassifier(weights='distance', n_jobs=-1, n_neighbors=1)

    def fit(self, train_data, train_lbls):
        self.clf.fit(train_data, train_lbls)
        return self

    def predict(self, test_data, test_lbls):
        classification = self.clf.predict(test_data)

        try:
            score = accuracy_score(test_lbls, classification)
        except ValueError:
            score = None
        return classification, score

# Back Propegation Perceptron
class BP_Perceptron:
    def __init__(self):
        self.label_offset = 0
        self.W = np.zeros(1)

    def fit(self, train_data, train_lbls, eta=1, eta_decay=0.01, max_iter=1000, annealing=True):
        # Create set of training classes
        classes = np.unique(train_lbls)
        class_count = len(classes)

        # Convert samples to float for faster numpy processing
        train_data = train_data.astype(float)

        # Augment data with bias to simplify linear discriminant function
        X = add_dummy_feature(train_data).transpose()
        n_features, n_samples = X.shape

        # Initialize weight matrix to random values
        self.W = np.random.rand(class_count, n_features)

        # Initialize labels for OVR (One vs Rest) binary classification
        ovr_lbls = np.where(train_lbls[np.newaxis, :] == classes[:, np.newaxis], 1, -1)

        for t in range(max_iter):
            # Evaluate perceptron criterion function
            F = np.multiply(ovr_lbls, np.dot(self.W, X))

            # Create set of misclassified samples (1 = misclassification, 0 = good classification)
            Chi = np.array(np.where(F <= 0,1 ,0)).astype(float)

            # Evaluate stopping criterion => no errors = stop
            Chi_sum = np.sum(Chi, axis=1)
            if np.count_nonzero(Chi_sum)==0:
                break

            # Calculate the delta summation of all misclassified samples
            delta = np.multiply(Chi,ovr_lbls).dot(X.transpose())

            # Exponential decay of epsilon
            if annealing:
                anneal = np.exp(-eta_decay*t)
            else:
                anneal = 1

            # Update W
            self.W += eta*anneal*delta

        return self

    def predict(self, test_data, test_label):
        return perceptron_classify(self.W, test_data, test_label)

# Mean-Square-Error Perceptron
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

    def predict(self, test_data, test_label):
        return perceptron_classify(self.W, test_data, test_label)

def perceptron_classify(W, test_data, test_lbls):
    # Create set of training classes
    classes = np.unique(test_lbls)
    label_offset = classes[0]
    test_data = test_data.astype(float)

    # Augment data with bias to simplify linear discriminant function
    X = add_dummy_feature(test_data).transpose()

    decision = np.dot(W,X)
    classification = np.argmax(decision,axis=0)+label_offset

    try:
        score = accuracy_score(test_lbls, classification)
    except ValueError:
        score = None
    return classification, score

#load data
train_data, train_label, test_data, test_label = loadData()

# nc = NC()
# nsc = NSC(subclass_count=2)
# nn = NN()
# bp = BP_Perceptron()
mse = MSE_Perceptron()

#data classification
# nc.fit(train_data, train_label)
# nc_classification, nc_score = nc.predict(test_data, test_label)

# nsc.fit(train_data, train_label)
# nsc_classification, nsc_score = nsc.predict(test_data, test_label)

# nn.fit(train_data, train_label)
# nn_classification, nn_score = nn.predict(test_data, test_label)

# bp.fit(train_data, train_label)
# bp_classification, bp_score = bp.predict(test_data, test_label)

mse.fit(train_data, train_label, 100)
mse_classification, mse_score = mse.predict(test_data, test_label)

# print(f"NC classification = {nc_classification}, Score = {nc_score}")
# print(f"NSC classification = {nsc_classification}, Score = {nsc_score}")
# print(f"NN classification = {nn_classification}, Score = {nn_score}")
# print(f"BP classification = {bp_classification}, Score = {bp_score}")
print(f"MSE classification = {mse_classification}, Score = {mse_score}")

filename = 'hejmeddig.sav'
pickle.dump(mse, open(filename, 'wb'))
