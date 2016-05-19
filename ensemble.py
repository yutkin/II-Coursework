import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

train_data = pd.read_csv('full_train_table.csv').as_matrix()
test_data = pd.read_csv('full_test_table.csv').as_matrix()

# Training set
X, y = train_data[:,:-1], train_data[:,-1]
# Testing set
Xt, yt = test_data[:,:-1], test_data[:,-1]

# Trying SVM
svm = SVC()
svm.fit(X,y)
svm.score(Xt, yt) * 100
>>> 94.170000000000002

# Trying Extra Trees
etc = ExtraTreesClassifier(n_estimators=1500, max_depth=None, 
                           max_features=4, min_samples_split=1)
etc.fit(X,y)
etc.score(Xt, yt) * 100
>>> 94.210000000000008

# Write ensemble to filesystem
file_path = open('extraTreesEnsemble.pickle', 'wb')
pickle.dump(etc, file_path)
file_path.close()
