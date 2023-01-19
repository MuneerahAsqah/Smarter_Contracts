from connections import *
#loading important libraires
import pandas as pd
import numpy as np

#import important librairies from skitlearn
from sklearn.svm import SVC #the classifier class, suppoer vector machine
from sklearn.model_selection import train_test_split #the library to split the data
from sklearn.preprocessing import StandardScaler #to scale the data, make all the values similar
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #for measuring the model performance
import json

#numpy randomised seed
np.random.seed(42)


def trainSVM(datafile, para=False):
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

    #create model instance
    svc = SVC(random_state=0, kernel='rbf')
    #train the model
    svc.fit(X_train, y_train)
    
    if(para):
        return scalar.mean_, scalar.var_, svc.support_vectors_, svc._dual_coef_,svc._intercept_,svc._gamma



def setParameters(m,v,sv,df,i,g):
    d = 10000000000
    #prepare scalar values to be written on-chain
    means = m.tolist()
    vars = v.tolist()
    #transform the float values to int
    means = [int(i*d) for i in means]
    vars = [int(i*d) for i in vars]
    #call the smart contract
    modelHandlerSVM.functions.setScalar(means,vars).transact()

    #the support vectors loop
    def setSupportVectors(sVectors):
        for i in range(sVectors.shape[0]):
            sv = sVectors[i].tolist()
            sv = [int(i*d) for i in sv]
            modelHandlerSVM.functions.setSupportVector(sv,i).transact()
    
    #prepare decison function parameters
    dual_coef = df[0].tolist()
    dual_coef = [int(i*d) for i in dual_coef]
    intercept = int(i[0]*d)
    gamma = int(g*d)
    
    #call the smart contract
    modelHandlerSVM.functions.setSVM(intercept,gamma).transact()
    modelHandlerSVM.functions.setDualCo(dual_coef).transact()
    
    #set the support vectors through the loop
    #setSupportVectors(sv)