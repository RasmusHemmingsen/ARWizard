import numpy as np
import pandas as pd
from Classifiers import NC, NSC, SVM, NN, BP_Perceptron, MSE_Perceptron
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

svm.fit(train_data, train_label)
nc.fit(train_data, train_label)
nsc.fit(train_data, train_label)
nn.fit(train_data, train_label)
bpp.fit(train_data, train_label)
mse.fit(train_data, train_label, epsilon=100)

svmPredictions = svm.predict(test_data)
ncPredictions = nc.predict(test_data)
nscPredictions = nsc.predict(test_data)
nnPredictions = nn.predict(test_data)
bppPredictions = bpp.predict(test_data)
msePredictions = mse.predict(test_data)

svmScore = accuracy_score(test_label, svmPredictions)
ncScore = accuracy_score(test_label, ncPredictions)
nscScore = accuracy_score(test_label, nscPredictions)
nnScore = accuracy_score(test_label, nnPredictions)
bppScore = accuracy_score(test_label, bppPredictions)
mseScore = accuracy_score(test_label, msePredictions)

print(f"svm score: {svmScore*100}%")
print(f"nc score: {ncScore*100}%")
print(f"nsc score: {nscScore*100}%")
print(f"nn score: {nnScore*100}%")
print(f"bpp score: {bppScore*100}%")
print(f"mse score: {mseScore*100}%")

filename = 'models/svmv1.sav'
pickle.dump(svm, open(filename, 'wb'))

filename = 'models/nc1.sav'
pickle.dump(nc, open(filename, 'wb'))
filename = 'models/nsc1.sav'
pickle.dump(nsc, open(filename, 'wb'))

filename = 'models/nnv1.sav'
pickle.dump(nn, open(filename, 'wb'))

filename = 'models/bppv1.sav'
pickle.dump(bpp, open(filename, 'wb'))

filename = 'models/msev1.sav'
pickle.dump(mse, open(filename, 'wb'))