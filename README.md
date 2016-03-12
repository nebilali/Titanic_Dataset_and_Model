# Titanic_Dataset_and_Model
Create model to predict who survived the Titanic. 

The Titanic DataSet contains information about each passenger and wheather they survived or not. Code is broken up into cleaning Data, Training, and outputing prediction. 

All code is in TitanicPredictions.py, and the boosted decision tree learner is written in boostedDT.py

The purpose of the cleanData function is to handle all missing inputs and create all features to be used in the model. 

	Cleaning the data:
		-generate gender feature -> male=1, female=0
		-generate embarked feature
			-fill in missing ports with most populare one
		-generate age feature
		  -fill in missing age of passangers with median age given passanger class
		-generate fare feature
		  -fill in missing age of passangers with median age given passanger class
		-generate fare range feature
		  -put ticket fare into buckets based on distribution
		-generate CabinID feature: 
		  -group based on section of cabin location
		-generate Title feature: 
		  -extract passanger's title from name
		-generate isChild feature: 
		  -is passanger younger than 15?
  
Best results where got using an ensamble learner which selects the most popular output with Random Forest, SVM, Logistic Regression, and self-written Boosted Decision Tree algorithms. 


