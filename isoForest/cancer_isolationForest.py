# Jacob uses Isolation forest and got the best results.. we will copy it.
# https://github.com/fkandah/2019-Behavioral-Model-Thesis/blob/master/run.py
#	His method for isolation forest is called: clfIsolationForest
#import AnomalyDetection
import time
import numpy as np
import pandas as pd
import sklearn
# from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
# from sklearn.model_selection import train_test_split

# ToDo: Change the argument values to something meaningful.. 
ml_model = IsolationForest(max_samples=1000, random_state=np.random.RandomState(42), n_jobs=-1, n_estimators=10000)#contamination=0.003

def main():
	# training data is the data that is going to be cleaned to remove anomalies while testing is entire file.
    training_df = pd.read_csv("../cancer_og.csv")
    testing_df = pd.read_csv("../cancer_og.csv")

    # Removing the anomalies from the csv.
    training_df = removeAnomalies(training_df)

    # These data df's are just the df but we removed the results because thats what the ml should do.
    training_data = training_df.drop('diagnosis', axis=1)
    testing_data = testing_df.drop('diagnosis', axis=1)

    # grab random subset (slow otherwise)
    #training_df = training_df.sample(len(training_df))

    # Train the model, using the anomaly free data.
    fitModel(training_data)

    # Lets pass in our testing data see if we can actually get an outlier/inlier..
    nd_results = ml_model.predict(testing_data)

    # getting true positives, false negatives, etc. 
    calcScore(nd_results, testing_df['diagnosis'])






def calcScore(ml_results, file_data):
	tp, fp, fn, tn = 0, 0, 0, 0

	for i in range(len(ml_results)):
		prediction = ml_results[i]
		was_anomaly = file_data[i]

		# prediction == 1 if inlier... -1 if outlier?? 
		# was_anomaly == 1 if inlier.. 0 if outlier. 

		# True positive..
		if(prediction == 1 and was_anomaly == 1):
			tp = tp+1
			continue
		# false positive... not great but we do want to minimize this.
		if(prediction == 1 and was_anomaly == 0):
			fp = fp+1
			continue
		# false negative... this is the big one that we want to minimize... 
		if(prediction == -1 and was_anomaly == 1):
			fn = fn+1
			continue
		# true negative... this is very good. 
		if(prediction == -1 and was_anomaly == 0):
			tn = tn+1
			continue

	printResults(tp, fp, fn, tn)



def fitModel(training_data):
	print("Fitting the model.")
	fit_start = time.time()
	ml_model.fit(training_data)
	fit_end = time.time()
	print("Time to fit: " + str(fit_end-fit_start))



def printResults(tp, fp, fn, tn):
	print("TP = ", tp)
	print("FP = ", fp)
	print("FN = ", fn)
	print("TN = ", tn)

	print("TPR = ", tp/(tp + fn))	# How many were true that we predicted.
	print("FPR = ", fp/(fp + tn))	# How many were false that we preicited. 

# Removes data from the csv where the diagnosis is 0... then resets the index of the df.
def removeAnomalies(input_df):
	input_df = input_df[input_df.diagnosis == 1]
	input_df = input_df.reset_index(drop=True)

	return input_df

main()

