""" Titanic Dataset Predictions.
Author : Nebil Ali
Date : 12rd March 2016

"""
import re
import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from boostedDT import *

#########################################
# Data cleanup
#########################################
cabinMap = {}
title_mapping = {}
Ports_dict = {}

def cleanData(titanic_df, isTestData=False):
	'''
	Desc:
		Preprocesses dataframe and returns clean Frame.
		Must run on training set before setting isTestData
			flag to True.
	Arguments:
		titanic_df is a pandas dataframe from titanic dataset
		isTestData, Boolean which idicates weather the  input
			dataframe is from training or testing set.

	'''

	titanic_df['Gender'] = titanic_df['Sex'].map({'female': 0, 'male': 1}).astype(int)
	
	# All missing Embarked -> just make them embark from most common place
	if len(titanic_df.Embarked[titanic_df.Embarked.isnull()]) > 0:
		titanic_df.Embarked[titanic_df.Embarked.isnull()] = titanic_df.Embarked.dropna().mode().values

	global Ports_dict
	if not isTestData:
		Ports = list(enumerate(np.unique(titanic_df['Embarked'])))    # determine all values of Embarked,
		Ports_dict = {name: i for i, name in Ports}              # set up a dictionary in the form  Ports : index
	
	titanic_df.Embarked = titanic_df.Embarked.map(lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

	# All the ages with no data -> make the median of all Ages
	if len(titanic_df.Age[titanic_df.Age.isnull()]) > 0:
		median_age = np.zeros(3)
		for f in range(0, 3):
			median_age[f] = titanic_df[titanic_df.Pclass == f+1]['Age'].dropna().median()
		for f in range(0, 3):
			titanic_df.loc[(titanic_df.Age.isnull()) & (titanic_df.Pclass == f+1), 'Age'] = median_age[f]

	# All the missing Fares -> assume median of their respective class
	if len(test_df.Fare[test_df.Fare.isnull()]) > 0:
		median_fare = np.zeros(3)
		for f in range(0, 3):                                              # loop 0 to 2
			median_fare[f] = test_df[test_df.Pclass == f+1]['Fare'].dropna().median()
		for f in range(0, 3):                                              # loop 0 to 2
			test_df.loc[(test_df.Fare.isnull()) & (test_df.Pclass == f+1), 'Fare'] = median_fare[f]

	# Cabin Feature -> Get Cabin Section
	def get_cabin(cabinID):
		return cabinID[0]

	titanic_df.loc[titanic_df.Cabin.isnull(), 'Cabin'] = 'Z'
	cabins = titanic_df["Cabin"].apply(get_cabin)
	global cabinMap
	if not isTestData:
		cabinMap = {k: i for i, k in enumerate(cabins)}
	else:
		for c in set(cabins):
			if c not in cabinMap:
				cabinMap[c] = cabinMap['Z']

	for k, v in cabinMap.items():
		cabins[cabins == k] = v
	titanic_df['newCabin'] = cabins

	# Title Feature -> Extract title from name
	def get_title(name):
		title_search = re.search(' ([A-Za-z]+)\.', name)
		if title_search:
			return title_search.group(1)
		return ""

	titles = titanic_df["Name"].apply(get_title)
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
	if  isTestData: 
		for title in set(titles): 				# Deal with new titles
			if title not in title_mapping:
				title_mapping[title] = 11
	
	for k, v in title_mapping.items():			# Assign Titles
		titles[titles == k] = v
	titanic_df["Title"] = titles

	# isChild Feature
	def isChild(age):
		if age < 15:
			return 1
		else:
			return 0

	titanic_df['isChild'] = titanic_df['Age'].map(isChild).astype(int)

	# Fare Range Feature
	def assignFareRange(fare):
		if fare > 93:
			return 0
		elif fare > 60:
			return 1
		elif fare > 26:
			return 2
		elif fare > 15:
			return 3
		elif fare > 8.05:
			return 4
		elif fare > 7.75:
			return 5
		else:
			return 6
		
	titanic_df['fareRange'] = titanic_df['Fare'].map(assignFareRange).astype(int)

	# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
	titanic_df = titanic_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
	titanic_df = titanic_df.drop(['SibSp', 'Parch', 'Age'], axis=1)

	return titanic_df

print 'Preprocessing Data...'
train_df = pd.read_csv('train.csv', header=0)     	# Load the train file into a dataframe
test_df = pd.read_csv('test.csv', header=0)       	# Load the test file into a dataframe
ids = test_df['PassengerId'].values
train_df = cleanData(train_df)
test_df = cleanData(test_df, isTestData=True)

X_train = train_df.values
Y_train = X_train[0::, 0]
X_train = X_train[0::, 1::]
X_test = test_df.values


##############################################################################
# Make Output

print 'Training Model...'
forest = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=3)
logReg = LogisticRegression()
clf = SVC(kernel='poly', degree=5)
btd = BoostedDT()

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

forest.fit(X_train, Y_train)
clf.fit(X_train, Y_train)
logReg.fit(X_train, Y_train)
btd.fit(X_train, Y_train)


print 'Predicting output...'
X_test = scaler.transform(X_test)

forestPred = forest.predict(X_test).astype(int)
clfPred = clf.predict(X_test).astype(int)
logRegPred = logReg.predict(X_test).astype(int)
btdPred = btd.predict(X_test).astype(int)

print 'Saving output to file...'
output = (clfPred + forestPred + logRegPred + btdPred) / 3

predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
