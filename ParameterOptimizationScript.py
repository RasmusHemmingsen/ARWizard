import numpy as np
import pandas as pd
from Classifiers import NC, NSC, SVM, NN, BP_Perceptron, MSE_Perceptron, DNN
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

def svmHyperparameterOptimization(train_data, test_data, train_label, test_label):
    # SVM decision function
    decision_function_shape_list = ["ovo","ovr"]
    for decision_function_shape in decision_function_shape_list:
        svm = SVM(decision_function_shape=decision_function_shape)
        svm.fit(train_data, train_label)
        svmPredictions, svmConfidence = svm.predict(test_data)
        svmScore = accuracy_score(test_label, svmPredictions)
        print(f"svm {decision_function_shape} score: {svmScore*100}%")

    # SVM kernel function
    kernel_list = ["linear","poly","rbf","sigmoid"]
    for kernel in kernel_list:
        svm = SVM(kernel=kernel)
        svm.fit(train_data, train_label)
        svmPredictions, svmConfidence = svm.predict(test_data)
        svmScore = accuracy_score(test_label, svmPredictions)
        print(f"svm {kernel} score: {svmScore*100}%")

    # SVM C parameter
    C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for C in C_list:
        svm = SVM(C=C)
        svm.fit(train_data, train_label)
        svmPredictions, svmConfidence = svm.predict(test_data)
        svmScore = accuracy_score(test_label, svmPredictions)
        print(f"svm C value {C} score: {svmScore*100}%")

def nnHyperparameterOptimization(train_data, test_data, train_label, test_label):
    neighborCounts = [1,2,3,4,5,6,7,8,9,10]
    for neighborCount in neighborCounts:
        nn = NN(n_neighbors=neighborCount)
        nn.fit(train_data, train_label)
        nnPredictions, nnConfidence = nn.predict(test_data)
        nnScore = accuracy_score(test_label, nnPredictions)
        print(f"nn {neighborCount} score: {nnScore*100}%")
    algorithms = ['auto', 'brute', 'kd_tree', 'ball_tree']
    for algorithm in algorithms:
        nn = NN(algorithm=algorithm)
        nn.fit(train_data, train_label)
        nnPredictions, nnConfidence = nn.predict(test_data)
        nnScore = accuracy_score(test_label, nnPredictions)
        print(f"nn algorithm {algorithm} score: {nnScore*100}%")
    weights = ['uniform', 'distance']
    for weight in weights:
        nn = NN(weight=weight)
        nn.fit(train_data, train_label)
        nnPredictions, nnConfidence = nn.predict(test_data)
        nnScore = accuracy_score(test_label, nnPredictions)
        print(f"nn weight {weight} score: {nnScore*100}%")

def bppHyperparameterOptimization(train_data, test_data, train_label, test_label):
    learning_rates = [0.1, 0.25, 0.5, 1, 5, 10, 20]
    for learning_rate in learning_rates:
        bpp = BP_Perceptron(eta=learning_rate, annealing=False)
        bpp.fit(train_data, train_label)
        bppPredictions, bppConfidence = bpp.predict(test_data)
        bppScore = accuracy_score(test_label, bppPredictions)
        print(f"bpp learning rate {learning_rate} score: {bppScore*100}%")
        bppScore = scoreWithMinConfidence(test_label, bppPredictions, bppConfidence, 0.7)
        print(f"bpp learning rate {learning_rate} score over 70% confidence: {bppScore*100}%")
        bppScore = scoreWithMinConfidence(test_label, bppPredictions, bppConfidence, 0.9)
        print(f"bpp learning rate {learning_rate} score over 90% confidence: {bppScore*100}%")
    eta_decays = [0.001, 0.0025, 0.005, 0.01, 0.05, 0.1, 0.5]
    for eta_decay in eta_decays:
        bpp = BP_Perceptron(eta_decay=eta_decay)
        bpp.fit(train_data, train_label)
        bppPredictions, bppConfidence = bpp.predict(test_data)
        bppScore = accuracy_score(test_label, bppPredictions)
        print(f"bpp decay {eta_decay} score: {bppScore*100}%")
        bppScore = scoreWithMinConfidence(test_label, bppPredictions, bppConfidence, 0.7)
        print(f"bpp decay {eta_decay} score over 70% confidence: {bppScore*100}%")
        bppScore = scoreWithMinConfidence(test_label, bppPredictions, bppConfidence, 0.9)
        print(f"bpp decay {eta_decay} score over 90% confidence: {bppScore*100}%")

def mseHyperparameterOptimization(train_data, test_data, train_label, test_label):
    epsilons = [0.0001 ,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    for epsilon in epsilons:
        mse = MSE_Perceptron(epsilon=epsilon)
        mse.fit(train_data, train_label)
        msePredictions, mseConfidence = mse.predict(test_data)
        # mseScore = accuracy_score(test_label, msePredictions)
        # print(f"mse epsilon {epsilon} score: {mseScore*100}%")
        # mseScore = scoreWithMinConfidence(test_label, msePredictions, mseConfidence, 0.7)
        # print(f"mse epsilon {epsilon} score over 70% confidence: {mseScore*100}%")
        mseScore = scoreWithMinConfidence(test_label, msePredictions, mseConfidence, 0.9)
        print(f"mse epsilon {epsilon} score over 90% confidence: {mseScore*100}%")

def dnnHyperparameterOptimization(train_data, test_data, train_label, test_label):
    dropouts = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    for dropout in dropouts:
        dnn = DNN(dropout=dropout)
        dnn.fit(train_data, train_label)
        dnnPredictions, dnnConfidence = dnn.predict(test_data)
        dnnScore = accuracy_score(test_label, dnnPredictions)
        print(f"dnn dropout {dropout} score: {dnnScore*100}%")
    
    sizes = [4560, 2280, 1140, 570, 285]
    for size in sizes:
        dnn = DNN(size=size)
        dnn.fit(train_data, train_label)
        dnnPredictions, dnnConfidence = dnn.predict(test_data)
        dnnScore = accuracy_score(test_label, dnnPredictions)
        print(f"dnn size {size} score: {dnnScore*100}%")
        dnnScore = scoreWithMinConfidence(test_label, dnnPredictions, dnnConfidence, 0.7)
        print(f"dnn size {size} score over 70% confidence: {dnnScore*100}%")
        dnnScore = scoreWithMinConfidence(test_label, dnnPredictions, dnnConfidence, 0.9)
        print(f"dnn size {size} score over 90% confidence: {dnnScore*100}%")

    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
    for learning_rate in learning_rates:
        dnn = DNN(learning_rate=learning_rate)
        dnn.fit(train_data, train_label)
        dnnPredictions, dnnConfidence = dnn.predict(test_data)
        dnnScore = accuracy_score(test_label, dnnPredictions)
        print(f"dnn learning_rate {learning_rate} score: {dnnScore*100}%")
        dnnScore = scoreWithMinConfidence(test_label, dnnPredictions, dnnConfidence, 0.7)
        print(f"dnn learning_rate {learning_rate} score over 70% confidence: {dnnScore*100}%")
        dnnScore = scoreWithMinConfidence(test_label, dnnPredictions, dnnConfidence, 0.9)
        print(f"dnn learning_rate {learning_rate} score over 90% confidence: {dnnScore*100}%")

train_data, test_data, train_label, test_label = loadData()

svmHyperparameterOptimization(train_data, test_data, train_label, test_label)
nnHyperparameterOptimization(train_data, test_data, train_label, test_label)
bppHyperparameterOptimization(train_data, test_data, train_label, test_label)
mseHyperparameterOptimization(train_data, test_data, train_label, test_label)
dnnHyperparameterOptimization(train_data, test_data, train_label, test_label)