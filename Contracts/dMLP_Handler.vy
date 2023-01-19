means: decimal[9]
var: decimal[9]
#the variables of MLP classifier
struct FirstWeight:
    fLayer: decimal[5]
struct SecondWeight:
    sLayer: decimal[2]
thirdWeight: decimal[2]
firstBias: decimal[5]
secondBias: decimal[2]
thirdBias: decimal
#mappings
first_weights: HashMap[int128, FirstWeight]
second_weights: HashMap[int128, SecondWeight]
#set the global value for converting
d: constant(decimal) = 10000000000.0

@external
def setScalar(_mean: int128[9], _var: int128[9]):
    #loop through, and convert the arrays
    for i in range(0,9):
        self.means[i] = convert(_mean[i],decimal)/d
        self.var[i] = convert(_var[i],decimal)/d

@external
def setFirstWeights(_mlp_w_1: int128[5], _c: int128):
    _firstWeights: decimal[5] = empty(decimal[5])
    for i in range(0,5):
        _firstWeights[i] = convert(_mlp_w_1[i],decimal)/d
    self.first_weights[_c] = FirstWeight({fLayer: _firstWeights})

@external
def setSecondWeights(_mlp_w_2: int128[2], _c: int128):
    _secondWeights: decimal[2] = empty(decimal[2])
    for i in range(0,2):
        _secondWeights[i] = convert(_mlp_w_2[i],decimal)/d
    self.second_weights[_c] = SecondWeight({sLayer: _secondWeights})
 
@external
def setThirdWeights(_mlp_w_3: int128[2]):
    for i in range(0,2):
        self.thirdWeight[i] = convert(_mlp_w_3[i],decimal)/d

@external
def setBias(_mlp_b_1: int128[5], _mlp_b_2: int128[2], _mlp_b_3: int128):
    for i in range(0,5):
        self.firstBias[i] = convert(_mlp_b_1[i],decimal)/d
    for i in range(0,2):
        self.secondBias[i] = convert(_mlp_b_2[i],decimal)/d
    self.thirdBias = convert(_mlp_b_3,decimal)/d
    
@internal
def scaleData(_vector: int128[9]) -> decimal[9]:
    #get the variance and get standart deviation by taking the square root
    _std: decimal[9] = self.var
    for i in range(0,9):
        _std[i] = sqrt(_std[i])
    #convert the new int vector into decimal, prepare for scaling
    _new_vector: decimal[9] = empty(decimal[9])
    for i in range(0,9):
        _new_vector[i] = convert(_vector[i],decimal)/d
    #scale the data using the standard deviation and means
    for i in range(0,9):
        _new_vector[i] = (_new_vector[i] - self.means[i])/_std[i]
    return _new_vector

@internal
def sumFirstNeuron(_input: decimal[9], _i: int128) -> decimal:
    _sum: decimal = 0.0
    for j in range(0,9):
        _weight: decimal[5] = self.first_weights[j].fLayer
        _aw: decimal = _input[j]*_weight[_i]
        _sum = _sum + _aw
    return _sum

@internal
def sumSecondNeuron(_input: decimal[5], _i: int128) -> decimal:
    _sum: decimal = 0.0
    for j in range(0,5):
        _weight: decimal[2] = self.second_weights[j].sLayer
        _bw: decimal = _input[j]*_weight[_i]
        _sum = _sum + _bw
    return _sum

@internal
def reLu(_x: decimal) -> decimal:
    return max(0.0,_x)

@external
def getPred(_datapoint: int128[9]) -> decimal:
    _scaledInput: decimal[9] = self.scaleData(_datapoint)
    _firstHidden: decimal[5] = empty(decimal[5])
    _secondHidden: decimal[2] = empty(decimal[2])
    _f: decimal = 0.0
    for i in range(0,5):
        _firstHidden[i] = self.reLu(self.sumFirstNeuron(_scaledInput,i)+self.firstBias[i])

    for i in range(0,2):
        _secondHidden[i] = self.reLu(self.sumSecondNeuron(_firstHidden,i)+self.secondBias[i])
    
    _sum: decimal = 0.0
    for i in range(0,2):
        _bw: decimal = _secondHidden[i]*self.thirdWeight[i]
        _sum = _sum + _bw
    _f = _sum + self.thirdBias
    return self.reLu(_f)
