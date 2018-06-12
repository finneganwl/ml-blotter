import numpy as np 
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import LabelBinarizer

def preprocess(df):
	# exlude "General/Other", it seems to be a mix of all the classifications instead of a separate category.
	df = df[df.category != 'General/Other']

	# select features
	X = np.array(df.iloc[:, 11:35])
	y = np.array(df.iloc[:, 35])

	#weight dimensions by number of variables used to represent that concept. 
	# e.g. since week is represented by 7 one hot variable, divide by 7.
	timeScaled = (1.0/2) * X[:,0:2]
	weekScaled = (1.0/7) * X[:,2:9]
	monthScaled = (1.0/12) * X[:,9:21]
	latLngScaled = (1.0/2) * X[:,21:23]
	dormScaled = (1.0/1) * X[:, 23:]

	X = np.concatenate((timeScaled, weekScaled, monthScaled, latLngScaled, dormScaled), axis=1)

	# 16 categories is a lot, try binary classification of one type vs all others
	# the '0' '1' is hacky but it solves string/int type mismatch, and makes LabelBinarizer set correct value to 1 for precision measure etc.
	y[(y != 'Drugs/Alcohol') & (y != 'Noise/Disturbances')] = '0'#'Something Else' #(y != 'Drugs/Alcohol') & (y != 'Theft') & (y != 'Suspicious Circumstances') & (y!= 'Property')
	y[(y == 'Drugs/Alcohol') | (y == 'Noise/Disturbances')] = '1' 

	lb = LabelBinarizer()
	y = np.array([number[0] for number in lb.fit_transform(y)])

	return [X, y]

if __name__ == "__main__":
	# import data
	df = pd.read_csv('data/train.csv', sep=',', header=0)

	X, y = preprocess(df)

	metric =  'f1' # 'precision' # 'recall' # 'accuracy' 

	for k in range (1,7):
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn, X, y, cv=10, scoring=metric)
		print "k = ", k, metric, " = ", np.mean(scores)

	k = 1
	y_pred = cross_val_predict(knn, X, y, cv=10)
	conf_mat = confusion_matrix(y ,y_pred)
	print "Confusion Matrix for k = 1:"
	print conf_mat

	logisticRegr = LogisticRegression()
	scores = cross_val_score(logisticRegr, X, y, cv=10, scoring=metric)
	print "Logistic Regression:", metric, " = ", np.mean(scores)

	dt = DecisionTreeClassifier()
	scores = cross_val_score(dt, X, y, cv=10, scoring=metric)
	print "Decision Tree:", metric, " = ", np.mean(scores)

	# compare to baseline
	if metric == 'accuracy':
		zeroR = DummyClassifier(strategy='most_frequent', random_state=42)
		scores = cross_val_score(zeroR, X, y, cv=10, scoring=metric)
		print "Baseline ZeroR:", metric, " = ", np.mean(scores)
	else:
		random = DummyClassifier(strategy='stratified', random_state=42)
		scores = cross_val_score(random, X, y, cv=10, scoring=metric)
		print "Baseline Random:", metric, " = ", np.mean(scores)