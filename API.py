from flask import Flask, request
from flask_api import status
import numpy as np
import pandas as pd
import json
from Classifiers import NC, NSC, SVM, NN, BP_Perceptron, MSE_Perceptron, ClassificationModel, DNN
app = Flask(__name__)

#Load the provided data from .mat files
def loadData():
    test_data = pd.read_csv(r"AR Wizard\Assets\TrainingData\test2.csv")

    return test_data.values

#test_data = loadData()

version = 4
svm = SVM().load(version)
nc = NC().load(version)
nsc = NSC(0).load(version)
nn = NN().load(version)
bpp = BP_Perceptron().load(version)
mse = MSE_Perceptron().load(version)
dnn = DNN().load(version)

@app.route('/features', methods=['POST'])
def post():
    method = request.args.get('method')
    test_data = np.array(json.loads(request.data))

    X = test_data.reshape(1,-1)

    if(method == "svm"): classification, percentage = svm.predict(X)
    elif(method == "nc"): classification, percentage = nc.predict(X)
    elif(method == "nsc"): classification, percentage = nsc.predict(X)
    elif(method == "nn"): classification, percentage = nn.predict(X)
    elif(method == "bpp"): classification, percentage = bpp.predict(X)
    elif(method == "mse"): classification, percentage = mse.predict(X)
    elif(method == "dnn"): classification, percentage = dnn.predict(X)
    else: return "Method not supported", status.HTTP_400_BAD_REQUEST

    model = ClassificationModel()

    model.Type = str(classification[0])
    model.Percentage = str(percentage[0])
    print(f"{model.Type} with {model.Percentage}% confidence")

    return json.dumps(model.__dict__)

app.run(host='0.0.0.0', port=5000)