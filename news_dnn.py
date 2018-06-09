
# coding: utf-8

# In[1]:



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

#Loading the data set - training data.
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

print 'Training...'

# In[14]:

# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
# We will be using the 'text_clf' going forward.
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

param_grid = {
	'clf__hidden_layer_sizes': [(10), (50), (100)]
}

pipe = Pipeline([
	('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('clf', MLPClassifier(
		max_iter=10,
		#solver='lbfgs',
		#hidden_layer_sizes=(50),
		verbose=True
	))
])

grid = GridSearchCV(pipe,
	param_grid=param_grid,
	cv=5,
	verbose=1,
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
