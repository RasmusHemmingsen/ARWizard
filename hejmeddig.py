from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import add_dummy_feature

#Load the provided data from .mat files
def loadData():
    #df = pd.read_csv(r"C:\Users\Nicklas\Dropbox\Computer Teknologi\CT 3. Semester\HandGestureDataGestureTogether.csv")
    train_data = np.genfromtxt(r"C:\Users\Nicklas\Dropbox\Computer Teknologi\CT 3. Semester\HandGestureDataGestureTogetherNoHead.csv", delimiter=',')
    train_label = np.array([0]*103 + [1]*98 + [2]*106)

    test_data = np.genfromtxt(r"C:\Users\Nicklas\Dropbox\Computer Teknologi\CT 3. Semester\HandGestureTestSetAndersNoHead.csv", delimiter=',')
    test_label = np.array([0]*9 + [1]*9 + [2]*10)

    return train_data, train_label, test_data, test_label

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


filename = 'hejmeddig.sav'
train_data, train_label, test_data, test_label = loadData()

mse = pickle.load(open(filename, 'rb'))

mse_classification, mse_score = mse.predict(test_data, test_label)
print(f"MSE classification = {mse_classification}, Score = {mse_score}")