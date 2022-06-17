from connections import *

v1 = [8,183,64,0,0,23.3,0.672,32,0]

v2 = [int(i*10000000000) for i in v1]

appropiate_gas = 300000000000

address = web3.toChecksumAddress("0x5b5D49Aa57334D4A09dcC317F4DF559dB2b3CCa8")

transaction_svm = modelHandlerSVM.functions.getPredictions(v2).buildTransaction({
	'gas': appropiate_gas,
	'nonce': web3.eth.get_transaction_count(address)
	})

transaction_mlp = modelHandlerMLP.functions.getPred(v2).buildTransaction({
	'gas': appropiate_gas,
	'nonce': web3.eth.get_transaction_count(address)
	})


svm_estimate = web3.eth.estimateGas(transaction_svm)
mlp_estimate = web3.eth.estimateGas(transaction_mlp)

print("The SVM estimate_gas is: ", svm_estimate)
print("The MLP estimate_gas is: ", mlp_estimate)

print("To wei, SVM: ", Web3.toWei(svm_estimate, "gwei"))
print("To wei, MLP: ", Web3.toWei(mlp_estimate, "gwei"))