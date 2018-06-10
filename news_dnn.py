# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

import numpy as np

#Loading the data set - training data.
dataset = fetch_20newsgroups(subset='all', remove=('headers','footers'), shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.20, random_state=0)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.20, random_state=0)


param_grid = {
	'vect__ngram_range': [(1,1), (1,2), (1,3)],
	#'vect__ngram_range': [(1,2)],
	'vect__min_df': [1, 3, 5]
	#'clf__hidden_layer_sizes': [(50)]#[(10), (50), (100)]
}

results = []

for ngram_range in [(1,3), (1,1), (1,2)]:
	for min_df in [1, 3, 5]:
		print("Training for {} and {}...".format(ngram_range, min_df))
		pipe = Pipeline([
			('vect', CountVectorizer(
				stop_words='english',
				ngram_range=ngram_range,
				min_df=min_df
			)),
			('tfidf', TfidfTransformer()),
			('clf', MLPClassifier(
				#max_iter=10,
				solver='adam',
				hidden_layer_sizes=(100),
				verbose=True,
				random_state=0,
				early_stopping=True,
				tol=0.001,
			))
		])
		classifier = pipe.fit(X_train, y_train)
		print "Testing ..."
		score = classifier.score(X_validate, y_validate)
		predicted = text_clf.predict(X_validate)
		accuracy =  accuracy_score(twenty_test.target, predicted)
		macrof1 = f1_score(y_validate, predicted, average='macro')  
		microf1 = f1_score(y_validate, predicted, average='micro') 
		results.append({
			params: (ngram_range, min_df),
			score: score,
			accuracy: accuracy,
			macrof1: macrof1,
			microf1: microf1
		})
		print "Got result:"
		print results[-1]


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
