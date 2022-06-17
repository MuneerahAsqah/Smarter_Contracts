from connections import *
from datasetHandling import *
from svm_modeling import *


#loads csv file
csvFile = r'diabetes.csv'
init_id = latest_id = 0

def writeDataset():
    if(web3.isConnected()):
        init_id = datasetHandler.functions.dCount().call()
        uploadDataset(csvFile)
        latest_id = datasetHandler.functions.dCount().call()
    else:
        raise Exception("Unable to write Dataset. Not connected to the web3 provider")

def readDataset():
    if(web3.isConnected()):
        latest_id = datasetHandler.functions.dCount().call()
        if(latest_id == 0):
            raise Exception("No data records")
        else:
            loadDataset(init_id, latest_id)
    else:
        raise Exception("Unable to read Dataset. Not connected to the web3 provider")

def fitSVM():
    if(web3.isConnected()):
        latest_id = datasetHandler.functions.dCount().call()
        if(latest_id == 0):
            raise Exception("No data records to train")
        else:
            #readDataset()
            loadedDataset = r'loadDataset.csv'
            trainSVM(loadDataset)
    else:
        raise Exception("Unable to load the Dataset and set SVM Model. Not connected to the web3 provider")
#how to get public variable
#latest_id = datasetHandeler.functions.dCount().call()

#to check connection with the network, prints true/false
print(web3.isConnected())
#to get latest block number
print(web3.eth.blockNumber)
