means: decimal[9]
var: decimal[9]
struct SupportVector:
    vector: decimal[9]
support_vectors: HashMap[int128, SupportVector]

svm_dualCoef: decimal[395]
svm_intercept: decimal
svm_gamma: decimal

#set the global value for converting
d: constant(decimal) = 10000000000.0

@external
def setScalar(_mean: int128[9], _var: int128[9]):
    #loop through, and convert the arrays
    for i in range(0,9):
        self.means[i] = convert(_mean[i],decimal)/d
        self.var[i] = convert(_var[i],decimal)/d
    
@external
def setSupportVector(_support_vector: int128[9], _id: int128):
    #increase the sv counter
    _decimal_vector: decimal[9] = empty(decimal[9])
    for i in range(0,9):
        _decimal_vector[i] = convert(_support_vector[i],decimal)/d
    self.support_vectors[_id] = SupportVector({vector: _decimal_vector})

@external
def setSVM(_intercept: int128, _gamma: int128):
    self.svm_intercept = convert(_intercept,decimal)/d
    self.svm_gamma = convert(_gamma,decimal)/d

@external
def setDualCo(_dual_coef: int128[395]):
    for i in range(0,395):
        self.svm_dualCoef[i] = convert(_dual_coef[i],decimal)/d

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
#calculate e^x according to taylor series
@internal
def taylor(_x: decimal) -> decimal:
    _term: decimal = 1.0
    _result: decimal = _term
    _n: decimal = 1.0
    for i in range(50):
        _term = (_term * _x) / _n
        _n += 1.0
        _result = _result + _term
    return _result

@internal
def fNorm(_x: decimal[9],_z: decimal[9]) -> decimal:
    _sum: decimal = 0.0 
    for i in range(0,9):
        _y: decimal = (_x[i]-_z[i])*(_x[i]-_z[i])
        _sum = _sum + _y
    return _sum

@internal
def rbf(_x: decimal[9], _v: decimal[9], _gamma: decimal) -> decimal:
    _norm: decimal = self.fNorm(_x,_v)
    _y: decimal = (_gamma*_norm)*(-1.0)
    return self.taylor(_y)

@external
def getPredictions(_datapoint: int128[9]) -> decimal:
    _scaled_datapoint: decimal[9] = self.scaleData(_datapoint)
    _rbfList: decimal[395] = empty(decimal[395])
    for i in range(0,395):
        _sv: decimal[9] = self.support_vectors[i].vector
        _rbfSV: decimal = self.rbf(_sv,_scaled_datapoint,self.svm_gamma)
        _rbfList[i] = _rbfSV
    _sum: decimal = 0.0
    for i in range(0,395):
        _y: decimal = self.svm_dualCoef[i]*_rbfList[i]
        _sum = _sum + _y
    return (_sum+self.svm_intercept)*(-1.0)