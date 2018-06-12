import numpy as np
#import pandas as pd 
import csv

with open("data/clean_features_blotter.csv", "r") as inputFile:
	reader = csv.reader(inputFile)
	headers = reader.next()
	print headers
	lst = []
	for row in reader:
		lst.append(row)


arr = np.array(lst)
np.random.shuffle(arr)

n = arr.shape[0]
splitN = int(.8*n)
print n
print splitN

train = arr[:splitN,:]
validation = arr[splitN:,:]

with open("data/validation.csv", "w") as validationFile:
	writer = csv.writer(validationFile)
	writer.writerow(headers)
	for row in validation:
		writer.writerow(row)

with open("data/train.csv", "w") as trainFile:
	writer = csv.writer(trainFile)
	writer.writerow(headers)
	for row in train:
		writer.writerow(row)

