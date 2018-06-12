import numpy as np 
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from modelSelection import preprocess 

# import data
df_train = pd.read_csv('data/train.csv', sep=',', header=0)
df_validation = pd.read_csv('data/validation.csv', sep=',', header=0)

X_train, y_train = preprocess(df_train)
X_validation, y_validation = preprocess(df_validation)
 
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_validation)

print "accuracy =", accuracy_score(y_validation, y_pred) 
print "f1 =", f1_score(y_validation, y_pred)
print "precision =", precision_score(y_validation, y_pred)
print "recall =", recall_score(y_validation, y_pred)

print "\n"
zeroR = DummyClassifier(strategy='most_frequent', random_state=42)
zeroR.fit(X_train, y_train)
y_pred = zeroR.predict(X_validation)
print "baseline zeroR accuracy =", accuracy_score(y_validation, y_pred) 

random = DummyClassifier(strategy='stratified', random_state=42)
random.fit(X_train, y_train)
y_pred = random.predict(X_validation)
print "baseline random f1 =", f1_score(y_validation, y_pred)
print "baseline random precision =", precision_score(y_validation, y_pred)
print "baseline random recall =", recall_score(y_validation, y_pred)
