# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import fetch_20newsgroups

import numpy as np
import pandas as pd

# Loading the training and testing data set
twenty_train = fetch_20newsgroups(subset='train', remove=('headers','footers'), shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', remove=('headers','footers'), shuffle=True)

print 'Training...'

# Define the hyperparameters that we want to try out in the grid search
param_grid = {
	'clf__hidden_layer_sizes': [(50), (100), (50, 50), (100, 100)],
	'clf__learning_rate_init': [1, 0.1, 0.01, 0.001, 0.0001]
}

# Setup a pipeline to define the preprocessing and feature extraction steps
pipe = Pipeline([
	('vect', CountVectorizer(
		stop_words='english'
	)),
	('tfidf', TfidfTransformer()),
	('clf', MLPClassifier(
		solver='adam',
		verbose=True,
		random_state=0,
		early_stopping=True,
		tol=0.001
	))
])

# Setup the parameters for the grid search
grid = GridSearchCV(pipe,
	param_grid=param_grid,
	cv=4,
	verbose=2,
	n_jobs=7
)

# Do the actual training and fit the classifier to the training set
text_clf = grid.fit(twenty_train.data, twenty_train.target)

print 'Done!'

# Print out information about the best classifier after training
print "Best cross-validation accuracy", grid.best_score_
print "Best parameters", grid.best_params_

results = pd.DataFrame(grid.cv_results_)
scores = np.array(results.mean_test_score).reshape(4, 5)
print "Scores for all parameters", scores

# Evaluate the best classifier based on the unseen test data
predicted = text_clf.predict(twenty_test.data)
print "Accuracy on test data:", accuracy_score(twenty_test.target, predicted)
print "Macro F1 on test data:", f1_score(twenty_test.target, predicted, average='macro')  
print "Micro F1 on test data:", f1_score(twenty_test.target, predicted, average='micro')
