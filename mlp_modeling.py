#from connections import *
#loading important libraires
import pandas as pd
import numpy as np
#import important librairies from skitlearn
from sklearn.neural_network import MLPClassifier #the classifier class, multi layaer perceptron
from sklearn.model_selection import train_test_split #the library to split the data
from sklearn.preprocessing import StandardScaler #to scale the data, make all the values similar
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #for measuring the model performance
import json

#numpy randomised seed
np.random.seed(42)

def trainNN(datafile, para=False):
    #reading the dataset .. call the loadDataset function in the future!
    dataset = pd.read_csv(datafile)
    #determine the dependent (y) and independent (x) variables
    X = dataset.drop(['RecordType'], axis=1)
    X = X.iloc[: , 1:]
    y = dataset['RecordType']
    #Now make them all as similar values
    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    #Splitting the dataset, test size is 20% or 10%
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)
    #train the NN networks
    NNclf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0, max_iter = 1000)
    NNclf.fit(X_train, y_train)
    #the prdiction score
    #y_predNN = NNclf.predict(X_test)
    #print the results
    #print("NN results:")
    #print(confusion_matrix(y_test,y_predNN))
    #print(classification_report(y_test,y_predNN))
    #print("Accuracy:", accuracy_score(y_test, y_predNN))

    if(para):
        return scalar.mean_, scalar.var_, NNclf.coefs_, NNclf.intercepts_

def setPara(m,v,weights,bias):
    d = 10000000000
    #prepare scalar values to be written on-chain
    means = m.tolist()
    vars = v.tolist()
    #transform the float values to int
    means = [int(i*d) for i in means]
    vars = [int(i*d) for i in vars]

    #call the smart contract
    #modelHandlerMLP.functions.setScalar(means,vars).transact()

    #prepare the weights, convert to int
    weights_1 = weights[0].tolist()
    for i in range(5):
        for j in range(9):
            weights_1[j][i] = int(weights_1[j][i]*d)
    weights_2 = weights[1].tolist()
    for i in range(2):
        for j in range(5):
            weights_2[j][i] = int(weights_2[j][i]*d)
    weights_3 = weights[2].tolist()
    for i in range(2):
        weights_3[i] = int(weights_3[i][0]*d)
    
    
    #set the weights
    for i in range(9):
        #modelHandlerMLP.functions.setFirstWeights(weights_1[i],i).transact()
    for i in range(5):
        #modelHandlerMLP.functions.setSecondWeights(weights_2[i],i).transact()
    
    #modelHandlerMLP.functions.setThirdWeights(weights_3).transact()
    #prepare the intercepts, convert to int
    bias_1 = [int(i*d) for i in bias[0]]
    bias_2 = [int(i*d) for i in bias[1]]
    bias_3 = int(bias[2][0]*d)
    #set the biases
    #modelHandlerMLP.functions.setBias(bias_1,bias_2,bias_3).transact()