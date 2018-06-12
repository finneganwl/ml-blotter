# ml-blotter 
This repository contains the code for the NU BLOTTER, a machine learning tool for predicting criminal activity near Northwestern University's Evanston Campus. Please see this [report](https://finneganwl.github.io/ml-blotter/) for more information.

<img width= 70% src="docs/img/time-sequence.gif">

## Developer Setup
### Requirements
* [Python 2.7+](https://www.python.org/downloads/)
* [pip](https://pip.pypa.io/en/stable/installing/)
* `pip install -r requirements.txt` should take care of the rest

### Files
* __scrapeBlotter.py__ scrapes blotter entries from the police department website.
* __generateFeatures.py__ converts the raw data into learnable features. _Note_: you will need a [Google Geocoding API Key](https://developers.google.com/maps/documentation/geocoding/get-api-key). Once you have the key, create a file in the root directory of this repo called config.py containing `GEOCODING_API_KEY = 'YourKeyHere'`.
* __categories.py__ maps incident type (~250 different crime descriptions from the raw data) to category (16 crime categories defined by me) used in generateFeatures.py
* __cleanAndScaleData.py__ corrects some common errors in the dataset and applies standard scaling to latitude and longitude.
* __separateValidationData.py__ splits the data into training and validation sets.
* __modelSelection.py__ tests a few machine learning models' performance on the training set.
* __reportValidationMetrics.py__ reports the final accuracy etc. of the chosen model on the validation set.
