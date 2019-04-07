from flask import Flask
from flask import request
import pickle
import numpy as np
from sklearn.preprocessing import add_dummy_feature
import json
app = Flask(__name__)

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


filename = r'C:\Users\Nicklas\Desktop\Source\ARWizard\hejmeddig.sav'
mse = pickle.load(open(filename, 'rb'))

@app.route('/postjson', methods=['POST'])
def post():
    test_data = np.array(json.loads(request.data))
        
    mse_classification = mse.predict(test_data.reshape(1,-1))
    print(f"{mse_classification}")
    return str(mse_classification[0])

app.run(host='0.0.0.0', port=5000)