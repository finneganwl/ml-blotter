import csv, json, requests, dateutil.parser, datetime, config, categories, numpy as np
# note: config = a file config.py you should make in this directory containing:
# GEOCODING_API_KEY = 'YourKey'

locations = {}

API_KEY = config.GEOCODING_API_KEY
ACTUALLY_QUERY = False # don't wanna waste those queries
def getCoords(location):
	# returns [lat, lng], [0, 0] if not found
	if ACTUALLY_QUERY and location != 'NA': #NA, Evanston is a valid address lol
		locQuery = "+".join(location.split())
		url = 'https://maps.googleapis.com/maps/api/geocode/json?address=' + locQuery + '+Evanston+IL&key=' + API_KEY
		res = requests.get(url)
		if res.status_code == requests.codes.ok:
			parsed = json.loads(res.text)
			if parsed['status'] == 'OK':
				lat = parsed['results'][0]['geometry']['location']['lat']
				lng = parsed['results'][0]['geometry']['location']['lng']
				return [lat, lng]
			else:
				print 'ERROR in Google Geocoding Response:'
				print parsed['status']
				print 'for query:', locQuery
		else:
			print 'Error fetching url:', url
	return [0, 0]



dorms = {}
with open('data/dorms.csv', 'r') as dormsFile:
	next(dormsFile)  # skip header line
	dormsReader = csv.reader(dormsFile)
	for row in dormsReader:
		dormAddr = row[0]
		dormLat = row[2]
		dormLng = row[3]
		dorms[dormAddr] = [dormLat, dormLng]

def getTimeInMinutes(dt):
	dtMidnight = datetime.datetime(dt.year, dt.month, dt.day, 0, 0)
	delta = dt - dtMidnight
	return delta.seconds / 60

def getTrigTime(timeInMinutes):
	# why do this:
	# https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/

	timeInMinutes = getTimeInMinutes(dt)
	minutesInDay = 60*24

	sin = np.sin(2*np.pi*timeInMinutes/minutesInDay)
	cos = np.cos(2*np.pi*timeInMinutes/minutesInDay)
	return [sin, cos]

def createDayOfWeekArray(dayOfWeek):
	out = []
	for i in range(7):
		if i == int(dayOfWeek):
			out.append(1)
		else:
			out.append(-1)
	return out

def createMonthArray(month):
	out = []
	for i in range(1,12+1): # month is 1-12 not 0-11
		if i == int(month):
			out.append(1)
		else:
			out.append(-1)
	return out


with open('data/raw_blotter.csv', 'r') as inFile:
	with open('data/features_blotter.csv', 'w') as outFile:
		reader = csv.reader(inFile)
		writer = csv.writer(outFile)
		headersOut = ["case_number", "date_reported", "date_occurred", "date_other", "address", "address_name", "incident_type", "criminal_offense", "disposition", "sin_time", "cos_time", "mon", "tues", "wed", "thurs", "fri", "sat", "sun", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "latitude", "longitude", "is_dorm", "category"]
		writer.writerow(headersOut)
		for row in reader:
			dt = dateutil.parser.parse(row[2].replace(':000',''), fuzzy = True)
			month = dt.month 
			year = dt.year 
			dayOfWeek = dt.weekday()
			sin_time, cos_time = getTrigTime(dt)

			location = row[4]
			isDorm = -1
			if location in dorms:
				isDorm = 1
				lat = dorms[location][0]
				lng = dorms[location][1]
				#print "thats a dorm", location
			else:
				# only query for location if not a dorm; we already have coordinates of dorms
				if location in locations:
					lat = locations[location]['lat']
					lng = locations[location]['lng']
				else:
					coords = getCoords(location)
					#numQueries += 1
					#print "numqueries = ", numQueries
					lat = coords[0]
					lng = coords[1]
					locations[location] = {'lat': lat, 'lng': lng} # saving locations reduces total number of queries (unique location names) to 1570

			category = categories.getCategory(row[6]) # get more general category for incident type, this is what we'll try to predict

			dayOfWeekArray = createDayOfWeekArray(dayOfWeek) # create one hot representations
			monthArray = createMonthArray(month)

			out = row + [sin_time, cos_time] + dayOfWeekArray + monthArray + [lat, lng, isDorm, category]

			#print out
			#wait = raw_input("wait")
			writer.writerow(out)