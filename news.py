# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import fetch_20newsgroups

import numpy as np

#Loading the data set - training data.
twenty_train = fetch_20newsgroups(subset='train', remove=('headers','footers'), shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', remove=('headers','footers'), shuffle=True)

print 'Training...'

# In[14]:

# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
# We will be using the 'text_clf' going forward.

param_grid = {
	'clf__hidden_layer_sizes': [(50), (100), (50, 50), (100, 100)],
	'clf__learning_rate_init': [1, 0.1, 0.01, 0.001, 0.0001]
}

pipe = Pipeline([
	('vect', CountVectorizer(
		stop_words='english'
	)),
	('tfidf', TfidfTransformer()),
	('clf', MLPClassifier(
		#max_iter=10,
		solver='adam',
		verbose=True,
		random_state=0,
		early_stopping=True,
		tol=0.001
	))
])

grid = GridSearchCV(pipe,
	param_grid=param_grid,
	cv=4,
	verbose=2,
	n_jobs=7
)

text_clf = grid.fit(twenty_train.data, twenty_train.target)

print 'Done!'

print "Best cross-validation accuracy", grid.best_score_
print "Test best score", grid.score(twenty_test.data, twenty_test.target)
print "Best parameters", grid.best_params_

# In[15]:

# Performance of NB Classifier
predicted = text_clf.predict(twenty_test.data)
print "Accuracy:", accuracy_score(twenty_test.target, predicted)
print "Macro F1:", f1_score(twenty_test.target, predicted, average='macro')  
print "Micro F1:", f1_score(twenty_test.target, predicted, average='micro') 