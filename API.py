from flask import Flask, request
import pickle
import numpy as np
import pandas as pd
import json
from Classifiers import NC, NSC, SVM, NN, BP_Perceptron, MSE_Perceptron, ClassificationModel
app = Flask(__name__)

#Load the provided data from .mat files
def loadData():
    test_data = pd.read_csv(r"AR Wizard\Assets\TrainingData\test2.csv")

    return test_data.values

test_data = loadData()

filename = r'models/modelv3.sav'

mse = pickle.load(open(filename, 'rb'))

@app.route('/postjson', methods=['POST'])
def post():
    #test_data = np.array(json.loads(request.data))

    classification, percentage = mse.predict(test_data.reshape(1,-1))
    model = ClassificationModel()

    model.Type = str(classification[0])
    model.Percentage = str(percentage[0])
    print(f"{model.Type} with {model.Percentage}% confidence")

    return json.dumps(model.__dict__)

app.run(host='0.0.0.0', port=5000)