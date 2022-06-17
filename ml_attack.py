import secml

import pandas as pd
import numpy as np

from sklearn.svm import SVC #the classifier class, suppoer vector machine
from sklearn.model_selection import train_test_split #the library to split the data
from sklearn.preprocessing import StandardScaler #to scale the data, make all the values similar
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #for measuring the model performance

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

parameters = {'kernel':('linear', 'rbf'), 'C':[1,2,3,4,5,6,7,8,9,10], 'gamma': 
              [0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5]}

#numpy randomised seed
np.random.seed(42)

#the datafile
datafile = r'corrupt_diabetes.csv'

#reading the dataset .. call the loadDataset function in the future!
dataset = pd.read_csv(datafile)
#determine the dependent (y) and independent (x) variables
X = dataset.drop(['RecordType'], axis=1)
X = X.iloc[: , 1:]
y = dataset['RecordType']

#Now make them all as similar values, Normalize
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

#Splitting the dataset, test size is 20% or 10%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

#create model instance
svc = SVC(random_state=0, kernel='rbf')
#train the model
svc.fit(X_train, y_train)

#measure the performance
y_pred = svc.predict(X_test)
print("SVM results:")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

x0 = X_test[5, :]
print(y_test[5,1])
y0 = y_test[5,1]  # Initial sample

noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
dmax = 0.4  # Maximum perturbation
lb, ub = 0, 1  # Bounds of the attack space. Can be set to `None` for unbounded
y_target = None  # None if `error-generic` or a class label for `error-specific`

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.3,
    'eta_min': 0.1,
    'eta_max': None,
    'max_iter': 100,
    'eps': 1e-4
}

from secml.adv.attacks.evasion import CAttackEvasionPGDLS
pgd_ls_attack = CAttackEvasionPGDLS(
    classifier=svc,
    double_init_ds=X_train,
    double_init=False,
    distance=noise_type,
    dmax=dmax,
    lb=lb, ub=ub,
    solver_params=solver_params,
    y_target=y_target)

# Run the evasion attack on x0
y_pred_pgdls, _, adv_ds_pgdls, _ = pgd_ls_attack.run(x0, y0)

print("Original x0 label: ", y0.item())
print("Adversarial example label (PGD-LS): ", y_pred_pgdls.item())

print("Number of classifier gradient evaluations: {:}"
      "".format(pgd_ls_attack.grad_eval))