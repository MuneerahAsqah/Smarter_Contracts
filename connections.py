from web3 import Web3
import json

#connection to the netwok
try:
    #network url
    ganache_url = 'http://127.0.0.1:7545'
    #setup the web3 provider to communicate with above netwrok, request_kwargs is to set the request timeout
    web3 = Web3(Web3.HTTPProvider(ganache_url, request_kwargs={'timeout': 240}))
except:
    print("Problems in connceting to the web3 provider. Check network inforamtion")

#setting up the defaultAccount
if(web3.isConnected()):
    #for calling the transact() function
    web3.eth.defaultAccount = web3.eth.accounts[3]
else:
    raise Exception("Unable to set defaultAccount. Not connected to the web3 provider")

#define the contracts
#CONTRACT 1: DATASETHANDELER
try:
    #a. get complied address
    address1 = web3.toChecksumAddress("0xb474abaE42c861BFAb8051ba959e9F3EE78A21ba")
    #b. load api info from remix
    abi1 = json.loads('[{"constant":true,"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"DataRecords","outputs":[{"internalType":"uint256","name":"id","type":"uint256"},{"internalType":"string","name":"record","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"dCount","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[{"internalType":"uint256","name":"_id","type":"uint256"}],"name":"readRecord","outputs":[{"internalType":"string","name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"internalType":"string","name":"_record","type":"string"}],"name":"writeRecord","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"}]')

    #c. assign the contract to a variable to call its functions
    datasetHandler = web3.eth.contract(address=address1, abi=abi1)
except:
    print("Problems in creating contact handler. Check web3 connection info")

#CONTRACT 2: MODELHANDLER
try:
    address = web3.toChecksumAddress("0xcCa88Abb239176CCBACb3d2ca66C4eB7EF7fe973")
    abi = json.loads('[{"constant":true,"inputs":[{"internalType":"uint256","name":"_id","type":"uint256"}],"name":"getSupportVector","outputs":[{"internalType":"string","name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"scalar","outputs":[{"internalType":"string","name":"means","type":"string"},{"internalType":"string","name":"vars","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"internalType":"string","name":"_dual_coef","type":"string"},{"internalType":"string","name":"_intercept","type":"string"},{"internalType":"string","name":"_gamma","type":"string"}],"name":"setSVM","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":false,"inputs":[{"internalType":"string","name":"_means","type":"string"},{"internalType":"string","name":"_vars","type":"string"}],"name":"setScalar","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":false,"inputs":[{"internalType":"string","name":"_support_vector","type":"string"}],"name":"setSupportVector","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"support_vectors","outputs":[{"internalType":"uint256","name":"id","type":"uint256"},{"internalType":"string","name":"support_vector","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"svCount","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"svm","outputs":[{"internalType":"string","name":"dual_coef","type":"string"},{"internalType":"string","name":"intercept","type":"string"},{"internalType":"string","name":"gamma","type":"string"}],"payable":false,"stateMutability":"view","type":"function"}]')


    modelHandler = web3.eth.contract(address=address, abi=abi)
except:
    print("Couldn't create model handler. Check web3 connection info")

#CONTRACT 2.1: SVM classifier modelHandler in vyper
try:
    address = web3.toChecksumAddress("0x4e37aC692E0CEC7E4F259A684FeCeD76B20faB89")
    abi = json.loads('[{"stateMutability":"nonpayable","type":"function","name":"setScalar","inputs":[{"name":"_mean","type":"int128[9]"},{"name":"_var","type":"int128[9]"}],"outputs":[],"gas":639070},{"stateMutability":"nonpayable","type":"function","name":"setSupportVector","inputs":[{"name":"_support_vector","type":"int128[9]"},{"name":"_id","type":"int128"}],"outputs":[],"gas":320717},{"stateMutability":"nonpayable","type":"function","name":"getSupportVector","inputs":[{"name":"_id","type":"int128"}],"outputs":[{"name":"","type":"fixed168x10[9]"}],"gas":20036},{"stateMutability":"nonpayable","type":"function","name":"setSVM","inputs":[{"name":"_intercept","type":"int128"},{"name":"_gamma","type":"int128"}],"outputs":[],"gas":71085},{"stateMutability":"nonpayable","type":"function","name":"setDualCo","inputs":[{"name":"_dual_coef","type":"int128[395]"}],"outputs":[],"gas":14033542},{"stateMutability":"nonpayable","type":"function","name":"getPredictions","inputs":[{"name":"_datapoint","type":"int128[9]"}],"outputs":[{"name":"","type":"fixed168x10"}],"gas":34634328},{"stateMutability":"view","type":"function","name":"rbfSum","inputs":[],"outputs":[{"name":"","type":"fixed168x10"}],"gas":2706},{"stateMutability":"view","type":"function","name":"rbf_counter","inputs":[],"outputs":[{"name":"","type":"int128"}],"gas":2736}]')
    modelHandlerSVM = web3.eth.contract(address=address, abi=abi)
except:
    print("Couldn't create model handler. Check web3 connection info")

#CONTRACT 2.2: MLP classifier modelHandler in vyper
try:
    address = web3.toChecksumAddress("0x880174375CC8a25AF05f189503B45FCB340dCcc0")
    abi = json.loads('[{"stateMutability":"nonpayable","type":"function","name":"setScalar","inputs":[{"name":"_mean","type":"int128[9]"},{"name":"_var","type":"int128[9]"}],"outputs":[],"gas":639070},{"stateMutability":"nonpayable","type":"function","name":"setFirstWeights","inputs":[{"name":"_mlp_w_1","type":"int128[5]"},{"name":"_c","type":"int128"}],"outputs":[],"gas":178517},{"stateMutability":"nonpayable","type":"function","name":"setSecondWeights","inputs":[{"name":"_mlp_w_2","type":"int128[2]"},{"name":"_c","type":"int128"}],"outputs":[],"gas":71897},{"stateMutability":"nonpayable","type":"function","name":"setThirdWeights","inputs":[{"name":"_mlp_w_3","type":"int128[2]"}],"outputs":[],"gas":71488},{"stateMutability":"nonpayable","type":"function","name":"setBias","inputs":[{"name":"_mlp_b_1","type":"int128[5]"},{"name":"_mlp_b_2","type":"int128[2]"},{"name":"_mlp_b_3","type":"int128"}],"outputs":[],"gas":284564},{"stateMutability":"nonpayable","type":"function","name":"getPred","inputs":[{"name":"_datapoint","type":"int128[9]"}],"outputs":[{"name":"","type":"fixed168x10"}],"gas":2324930}]')
    modelHandlerMLP = web3.eth.contract(address=address, abi=abi)
except:
    print("Couldn't create MLP model handler. Check web3 connection")
