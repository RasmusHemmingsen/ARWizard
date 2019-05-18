import numpy as np
import pandas as pd
from Classifiers import NC, NSC, SVM, NN, BP_Perceptron, MSE_Perceptron, DNN, DNN2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import confusion_matrix

def loadData():
    X = pd.read_csv(r"AR Wizard\Assets\TrainingData\HandGestureData3.csv")
    y = np.array([0]*131 + [1]*131 + [2]*125 + [3]*127 + [0]*393 + [1]*317 + [2]*448 + [3]*351)

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=7)

    return X_train, X_test, y_train, y_test


def scoreWithMinConfidence(test_labels, predictions, confidences, minConfidence):
    if(len(test_labels) != len(predictions) and len(test_labels) != len(confidences)):
        raise Exception('Labels, predictions and confidences should have same length')
    failed = 0
    success = 0
    for i in range(len(test_labels)):
        if confidences[i] < minConfidence:
            failed = failed + 1
        elif(test_labels[i] == predictions[i]):
            success = success + 1
        else:
            failed = failed + 1
    
    return success / (failed + success)


train_data, test_data, train_label, test_label = loadData()

svm = SVM(decision_function_shape="ovo", kernel="rbf", C=100)
nc = NC()
nsc = NSC(3)
nn = NN(n_neighbors=6, algorithm="auto", weight='distance')
bpp = BP_Perceptron(eta=0.5, annealing=False)
mse = MSE_Perceptron(epsilon=10000)
dnn = DNN(size=570)

svm.fit(train_data, train_label)
nc.fit(train_data, train_label)
nsc.fit(train_data, train_label)
nn.fit(train_data, train_label)
bpp.fit(train_data, train_label)
mse.fit(train_data, train_label)
dnn.fit(train_data, train_label)

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

confidences = [0.7, 0.8, 0.9, 0.95]
for confidence in confidences:
    svmminScore =  scoreWithMinConfidence(test_label, svmPredictions, svmConfidence, confidence)
    nnminScore =  scoreWithMinConfidence(test_label, nnPredictions, nnConfidence, confidence)
    bppminScore =  scoreWithMinConfidence(test_label, bppPredictions, bppConfidence, confidence)
    mseminScore =  scoreWithMinConfidence(test_label, msePredictions, mseConfidence, confidence)
    dnnminScore =  scoreWithMinConfidence(test_label, dnnPredictions, dnnConfidence, confidence)

    print(f"svm min {confidence*100}% score: {svmminScore*100}%")
    print(f"nn min {confidence*100}% score: {nnminScore*100}%")
    print(f"bpp min {confidence*100}% score: {bppminScore*100}%")
    print(f"mse min {confidence*100}% score: {mseminScore*100}%")
    print(f"dnn min {confidence*100}% score: {dnnminScore*100}%")

svmConfusionMatrix = confusion_matrix(test_label, svmPredictions)
ncConfusionMatrix = confusion_matrix(test_label, ncPredictions)
nscConfusionMatrix = confusion_matrix(test_label, nscPredictions)
nnConfusionMatrix = confusion_matrix(test_label, nnPredictions)
bppConfusionMatrix = confusion_matrix(test_label, bppPredictions)
mseConfusionMatrix = confusion_matrix(test_label, msePredictions)
dnnConfusionMatrix = confusion_matrix(test_label, dnnPredictions)
print("svm confusion matrix:")
print(svmConfusionMatrix)
print("nc confusion matrix:")
print(ncConfusionMatrix)
print("nsc confusion matrix:")
print(nscConfusionMatrix)
print("nn confusion matrix:")
print(nnConfusionMatrix)
print("bpp confusion matrix:")
print(bppConfusionMatrix)
print("mse confusion matrix:")
print(mseConfusionMatrix)
print("dnn confusion matrix:")
print(dnnConfusionMatrix)



version = 7
svm.save(version)
nc.save(version)
nsc.save(version)
nn.save(version)
bpp.save(version)
mse.save(version)
dnn.save(version)