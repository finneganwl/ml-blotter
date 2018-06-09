import numpy as np 
import pandas as pd

#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.tree import DecisionTreeClassifier

# import data
df = pd.read_csv('train.csv', sep=',', header=0)
#df = df[df.category != 'General/Other']

# select features
X = np.array(df.iloc[:, 11:35])
y = np.array(df.iloc[:, 35])
y[y != 'Drugs/Alcohol'] = '0'#'Something Else' #(y != 'Drugs/Alcohol') & (y != 'Theft') & (y != 'Suspicious Circumstances') & (y!= 'Property')
y[y == 'Drugs/Alcohol'] = '1' 
print y[0:100]

#weight dimensions by number of variables used to represent that concept. 
# e.g. since week is represented by 7 one hot variable, divide by 7.
timeScaled = (1.0/2) * X[:,0:2]
weekScaled = (1.0/7) * X[:,2:9]
monthScaled = (1.0/12) * X[:,9:21]
latLngScaled = (1.0/2) * X[:,21:23]

dormScaled = (1.0/1) * X[:, 23:]

X = np.concatenate((timeScaled, weekScaled, monthScaled, latLngScaled, dormScaled), axis=1)
#X = np.concatenate((X[:,:21], latLngScaled, X[:, 23:]), axis=1)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)

lb = preprocessing.LabelBinarizer()
y = np.array([number[0] for number in lb.fit_transform(y)])
#y[y==0] = 2
#y[y==1] = 0
#y[y==2] = 1
print y[0:100]
"""
for k in range (1,7):
	#k = 7
	knn = KNeighborsClassifier(n_neighbors=k)

	#knn.fit(X_, y_train)

	scores = cross_val_score(knn, X, y, cv=10, scoring='recall')
	print "k = ", k, "f1_weighted = ", np.mean(scores)
"""

dt = DecisionTreeClassifier(max_depth=3)
scores = cross_val_score(dt, X, y, cv=10, scoring='recall')
print "Decision Tree:", "f1_weighted = ", np.mean(scores)

#clf = dt.fit(X, y)

#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
from sklearn import tree
from graphviz import Source #import graphviz
#import pydotplus
#dot_data = StringIO()
#export_graphviz(clf, out_file=dot_data,  filled=True, rounded=True, special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())

clf = dt.fit(X, y)
dot_data = tree.export_graphviz(clf, out_file="Tree.dot", filled=True, rounded=True, special_characters=True) 
graph = Source(dot_data) 
#graph.format = 'png'
graph.render("X")


#pred = knn.predict(X_test)

#print accuracy_score(y_test, pred)