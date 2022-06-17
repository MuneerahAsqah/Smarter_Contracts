#from connections import *
#loading important libraires
import pandas as pd
import numpy as np
import timeit
#import important librairies from skitlearn
from sklearn.neural_network import MLPClassifier #the classifier class, multi layaer perceptron
from sklearn.model_selection import train_test_split #the library to split the data
from sklearn.preprocessing import StandardScaler #to scale the data, make all the values similar
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #for measuring the model performance
import json
import foolbox as fb

#numpy randomised seed
np.random.seed(42)
#the datafile
datafile = r'corrupt_diabetes.csv'

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
    v1 = [8,183,64,0,0,23.3,0.672,32,0]
    new_v1 = np.array(v1).reshape(1,-1)
    v2 = [int(i*10000000000) for i in v1]
    """
    MLPdec_time = []
    MLPdecVyper_time = []
    for i in range(10):
        start_time1 = timeit.default_timer()
        modelHandlerMLP.functions.getPred(v2).call()
        elapsed_time1 = timeit.default_timer() - start_time1
        print(elapsed_time1)
        MLPdecVyper_time.append(elapsed_time1)
    
    for i in range(10):
        start_time2 = timeit.default_timer()
        NNclf.predict(scalar.transform(new_v1))
        elapsed_time2 = timeit.default_timer() - start_time2
        print(elapsed_time2)
        MLPdec_time.append(elapsed_time2)
    
    def Average(list):
        return sum(list) / len(list)

    print('Average elapsed time for SC df: ', Average(MLPdecVyper_time))
    print('--------------------------------')
    print('Average elapsed time for built-in df: ', Average(MLPdec_time))
    """
    #scaledVector = scalar.transform(new_v1)
    #print("This classifier prediction: ", NNclf.predict(scaledVector))
    #return the classifier's parameters
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
    print('means: ', means)
    print('vars: ', vars)
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
        print("first_ weights of id [", i,'] is: ', weights_1[i])
        #modelHandlerMLP.functions.setFirstWeights(weights_1[i],i).transact()
    for i in range(5):
        print("second_ weights of id [", i,'] is: ', weights_2[i])
        #modelHandlerMLP.functions.setSecondWeights(weights_2[i],i).transact()
    
    #modelHandlerMLP.functions.setThirdWeights(weights_3).transact()
    print('third_weights are: ', weights_3)
    #start_time = timeit.default_timer()
    #prepare the intercepts, convert to int
    bias_1 = [int(i*d) for i in bias[0]]
    bias_2 = [int(i*d) for i in bias[1]]
    bias_3 = int(bias[2][0]*d)
    #set the biases
    #modelHandlerMLP.functions.setBias(bias_1,bias_2,bias_3).transact()
    print("First bias: ", bias_1)
    print("Second bias: ", bias_2)
    print("third bias: ", bias_3)
    #print('Time for setBias time: ', timeit.default_timer() - start_time)

def getPred():
    weights, bias, vector = trainNN(datafile, para=True)
    # the acttivation function
    def reLu(x):
        return max(0,x)
    # two empty lists for 2 hidden layers
    A = []
    B = []
    #the loop for summasion, i for 2 hidden layers + outcome, j for neurons, k for wieghts
    for i in range(3):
        if i == 0:
            for j in range(5):
                sum_1 = 0
                for k in range(8):
                    xw = vector[k]*weights[i][k][j]
                    sum_1 = sum_1+xw
                a = sum_1 + bias[i][j]
                a = reLu(a)
                A.append(a)
        elif i == 1:
            for j in range(2):
                sum_2 = 0
                for k in range(5):
                    aw = A[k]*weights[i][k][j]
                    sum_2 = sum_2+aw
                b = sum_2 + bias[i][j]
                b = reLu(b)
                B.append(b)
        elif i == 2:
            sum_3 = 0
            for k in range(2):
                bw = B[k]*weights[i][k]
                sum_3 = sum_3+bw
            f = sum_3 + bias[i]

    print("the result is: ", reLu(f))
    return reLu(f)
