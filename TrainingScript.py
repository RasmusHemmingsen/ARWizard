import numpy as np
import pandas as pd
from Classifiers import NC, NSC, SVM, NN, BP_Perceptron, MSE_Perceptron, DNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def loadData():
    X = pd.read_csv(r"AR Wizard\Assets\TrainingData\HandGestureData2.csv")
    y = np.array([0]*131 + [1]*131 + [2]*125 + [3]*127)

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=7)

    return X_train, X_test, y_train, y_test

train_data, test_data, train_label, test_label = loadData()

svm = SVM()
nc = NC()
nsc = NSC(3)
nn = NN()
bpp = BP_Perceptron()
mse = MSE_Perceptron()
dnn = DNN()

svm.fit(train_data, train_label)
nc.fit(train_data, train_label)
nsc.fit(train_data, train_label)
nn.fit(train_data, train_label)
bpp.fit(train_data, train_label)
mse.fit(train_data, train_label, epsilon=100)
dnn.fit(train_data, train_label, epochs=2)

svmPredictions, svmConfidence = svm.predict(test_data)
ncPredictions, ncConfidence = nc.predict(test_data)
nscPredictions, nscConfidence = nsc.predict(test_data)
nnPredictions, nnConfidence = nn.predict(test_data)
bppPredictions, bppConfidence = bpp.predict(test_data)
msePredictions, mseConfidence = mse.predict(test_data)
dnnPredictions, dnnConfidence = dnn.predict(test_data)

svmScore = accuracy_score(test_label, svmPredictions)
ncScore = accuracy_score(test_label, ncPredictions)
nscScore = accuracy_score(test_label, nscPredictions)
nnScore = accuracy_score(test_label, nnPredictions)
bppScore = accuracy_score(test_label, bppPredictions)
mseScore = accuracy_score(test_label, msePredictions)
dnnScore = accuracy_score(test_label, dnnPredictions)

print(f"svm score: {svmScore*100}%")
print(f"nc score: {ncScore*100}%")
print(f"nsc score: {nscScore*100}%")
print(f"nn score: {nnScore*100}%")
print(f"bpp score: {bppScore*100}%")
print(f"mse score: {mseScore*100}%")
print(f"dnn score: {dnnScore*100}%")

version = 1
svm.save(version)
nc.save(version)
nsc.save(version)
nn.save(version)
bpp.save(version)
mse.save(version)
dnn.save(version)