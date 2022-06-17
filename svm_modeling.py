from connections import *
#loading important libraires
import pandas as pd
import numpy as np
import timeit

#import important librairies from skitlearn
from sklearn.svm import SVC #the classifier class, suppoer vector machine
from sklearn.model_selection import train_test_split #the library to split the data
from sklearn.preprocessing import StandardScaler #to scale the data, make all the values similar
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #for measuring the model performance
import json

#numpy randomised seed
np.random.seed(42)

#for converting np arrays to json
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder,self).default(obj)

#the datafile
datafile = r'corrupt_diabetes.csv'

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

    #testing purpoeses
    v1 = [8,183,64,0,0,23.3,0.672,32,0]
    v2 = [int(i*10000000000) for i in v1]
    new_v1 = np.array(v1).reshape(1,-1)

    def SVMdec():
        svc._decision_function(scalar.transform(new_v1))

    def SVMdecVyper():
        modelHandlerSVM.functions.getPredictions(v2).call()
    #measure the performance
    #y_pred = svc.predict(X_test)
    #print("SVM results:")
    #print(confusion_matrix(y_test,y_pred))
    #print(classification_report(y_test,y_pred))
    #print("Accuracy:", accuracy_score(y_test, y_pred))
    #Measuring the performance
    
    SVMdec_time = []
    SVMdecVyper_time = []
    for i in range(10):
        start_time1 = timeit.default_timer()
        modelHandlerSVM.functions.getPredictions(v2).call()
        elapsed_time1 = timeit.default_timer() - start_time1
        print(elapsed_time1)
        SVMdecVyper_time.append(elapsed_time1)
    
    for i in range(10):
        start_time2 = timeit.default_timer()
        svc._decision_function(scalar.transform(new_v1))
        elapsed_time2 = timeit.default_timer() - start_time2
        print(elapsed_time2)
        SVMdec_time.append(elapsed_time2)
    
    def Average(list):
        return sum(list) / len(list)

    print('Average elapsed time for SC df: ', Average(SVMdecVyper_time))
    print('--------------------------------')
    print('Average elapsed time for built-in df: ', Average(SVMdec_time))


    #print("The built in df: ", svc._decision_function(scalar.transform(new_v1)))
    #print("My descision functions: ", getPredictions(v1))
    
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

    

def getPredictions(v):

    #get the scalar values
    means, vars = getScalarData()
    stds = np.sqrt(vars)
    #the scalar inner function
    def scale_data(array,means=means,stds=stds):
        return (array-means)/stds
    #get decison function values
    support_vectors, dual_coef, intercept, gamma = getSVMData()

    #The new data needs to be also scaled with htw same scaler
    #first convert it to array
    new_v = np.array(v).reshape(1,-1)
    #second scale it with the inner scaler function
    new_v_scaled = scale_data(new_v,means,stds)
    print(new_v_scaled)
    print(support_vectors[23])
    #to collect RBF
    def RBF(x,z,gamma,axis=None):
        e = np.exp(-gamma*np.linalg.norm(x-z, axis=axis)**2)
        return e

    #it is time for predictions
    A = []
    for x in support_vectors:
        A.append(RBF(x,new_v_scaled,gamma))
    A = np.array(A)
    sum = np.sum(dual_coef*A)
    ss = 0
    sss=0

    for i in range(0,350):
        y = dual_coef[0][i]*A[i]
        if(i==23):
            print("coef: ", dual_coef[0][i], " rbf: ", A[i], " multiplication :", y)
        sss = sss+y
    #retrun prediction value
    return (np.sum(dual_coef*A)+intercept)*(-1),A,ss,sss