import numpy as np 
import pandas as pd
import csv

from sklearn.preprocessing import scale

# rough borders of evanston, exclude all other locations
NORTHERN_BOUND = 42.071889
SOUTHERN_BOUND = 42.019479
EASTERN_BOUND = -87.665299
WESTERN_BOUND = -87.732457

# col numbers in input file
LAT = 30
LNG = 31
DERIVED_FEATURES_START = 9

df = pd.read_csv('features_blotter.csv', sep=',', header=0)
# also takes care of missing values encoded as [0,0]
df = df[df.latitude < NORTHERN_BOUND]
df = df[df.latitude > SOUTHERN_BOUND]
df = df[df.longitude < EASTERN_BOUND]
df = df[df.longitude > WESTERN_BOUND]

# sometimes these messy incident types 
# have useful information in the "criminal offense" column that would make it better classified as something else
# and actually represent a significant portion of alcohol and noise cases
messyIncidentTypes = ['Assist Other Agency', 'Service to Other Agency', 'Public Contact', 'Public Service']
corrections = {
	"Illegal Consumption by Minor": "Drugs/Alcohol",
	"Noise Prohibition (Loud Continued)": "Noise/Disturbances",
	"Disorderly House (Keeping A) Noise Violation": "Noise/Disturbances",
	"Drunkenness": "Drugs/Alcohol",
	"All  Disorderly Conduct": "Noise/Disturbances",
	"Noise Prohibition of Loud Yelling or Shouting 2300 to 0700": "Noise/Disturbances",
	"Alcoholic Beverages(Consumption/Possession In Public)": "Drugs/Alcohol",
	"Noise Prohibition of Loud Radio Type Music 2300 to 0700": "Noise/Disturbances",
	"Illegal Possession by Minor": "Drugs/Alcohol",
	"Underage Possession of Alcohol-sale,gift delivery/purchase or posess": "Drugs/Alcohol",
	"Noise Prohibition of Loud Speakers 2300 to 0700": "Noise/Disturbances",
	"Possession of Cannabis Prohibited": "Drugs/Alcohol",
	"Unlawful Use/Possession Drug Paraphernalia": "Drugs/Alcohol",
	"Possession of Cannabis 30 gm and Under": "Drugs/Alcohol"
}
for incid in messyIncidentTypes:
	for key, value in corrections.iteritems():
		selector = (df.incident_type == incid) & (df.criminal_offense == key)
		df.loc[selector,("category")] = value
		#print incid, key, sum(selector)



# use [LAT] in indexing so get obeject of shape (R,1) instead of (R,)
matrix = np.array(df) # convert to numpy form
#print matrix[1:10,[LAT]]
latScaled = scale(matrix[:,[LAT]].astype(float))
lngScaled = scale(matrix[:,[LNG]].astype(float))
#print latScaled[1:10]


# want to move latitude and longitude over near the raw features so easier for ML later
outMatrix = np.concatenate((matrix[:,0:DERIVED_FEATURES_START], matrix[:,[LAT]], matrix[:,[LNG]], matrix[:,DERIVED_FEATURES_START:LAT], latScaled, lngScaled, matrix[:,(LNG+1):]), axis=1)
print outMatrix[1,:]

with open('clean_features_blotter.csv', 'w') as outFile:
	writer = csv.writer(outFile)
	headersOut = ["case_number", "date_reported", "date_occurred", "date_other", "address", "address_name", "incident_type", "criminal_offense", "disposition", "latitude", "longitude", "sin_time", "cos_time", "mon", "tues", "wed", "thurs", "fri", "sat", "sun", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "latitude_scaled", "longitude_scaled", "is_dorm", "category"]
	writer.writerow(headersOut)
	numRows = outMatrix.shape[0]
	for i in range(numRows):
		row = outMatrix[i,:]
		writer.writerow(row)





