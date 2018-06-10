# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import fetch_20newsgroups



# Stemming Code

# import nltk
# nltk.download('stopwords')

# from nltk.stem.snowball import SnowballStemmer
# stemmer = SnowballStemmer("english", ignore_stopwords=True)

# class StemmedCountVectorizer(CountVectorizer):
#     def build_analyzer(self):
#         analyzer = super(StemmedCountVectorizer, self).build_analyzer()
#         return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

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
	'vect__ngram_range': [(1,1), (1,2), (1,3)],
	#'vect__ngram_range': [(1,2)],
	'vect__min_df': [1, 3, 5]
	#'clf__hidden_layer_sizes': [(50)]#[(10), (50), (100)]
}

pipe = Pipeline([
	('vect', CountVectorizer(
		stop_words='english'
	)),
	('tfidf', TfidfTransformer()),
	('clf', MLPClassifier(
		#max_iter=10,
		solver='adam',
		hidden_layer_sizes=(100),
		verbose=True,
		random_state=0,
		early_stopping=True,
		tol=0.001
	))
])

grid = GridSearchCV(pipe,
	param_grid=param_grid,
	cv=4,
	verbose=1,
	#n_jobs=4
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
