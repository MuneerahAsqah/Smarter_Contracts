import csv
import json
from connections import *
import timeit

#the function to upload the data records
def uploadDataset(csvFile):
    try:
        with open(csvFile, encoding='utf-8') as csvf:
            #load the csv parser to read through the csv file
            csvReader = csv.DictReader(csvf)
            #looping through each record/row and store on blockchain
            for row in csvReader:
                #convert to json string
                record = json.dumps(row)
                #call the addDataset functions
                #this is how to call a set function
                datasetHandler.functions.writeRecord(record).transact()
        #print confirmation
        print('dataset uploaded succefully')
    except:
        print('Unable to write dataset. Check contract connection info.')

#the function to convert json string record back to CSV
def jsonCSV(jsonList):
    try:
        #open a new file to write the records
        csvNewFile = open('loadedDataset.csv','w')
        #define and assign the writer
        csvWriter = csv.writer(csvNewFile)
        #define the keys (columns) from one of the records
        csvWriter.writerow(jsonList[0].keys())
        #loop through the record list and write the values
        for row in jsonList:
            csvWriter.writerow(row.values())
        #close the file
        csvNewFile.close()
        #print confirmation in the console
        print('Exporting of CSV file done')
    except:
        print('Unable to export CSV is file')

#define the function that loads the json records
def loadDataset(init_id, latest_id):
    #initialize the counters. i for reading through the record list,
    #j for calculating how many records in the info message
    j = init_id
    #initialize an array to hold the records and the variables
    recordList = []
    i = init_id
    try:
        #loop through the number of records
        while(i<=latest_id):
            #call the datasetHandeler and read the value
            #this is how to call a get function
            record = datasetHandler.functions.readRecord(i).call()
            #convert the string into a json string
            jsonString = json.loads(record)
            #append this json string to the records array
            recordList.append(jsonString)
            #increment the counter
            i+=1

        #call the function that convet json list to csv again
        jsonCSV(recordList)
    except:
            print('Unable to load the dataset')