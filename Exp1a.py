
# coding: utf-8

# In[1]:

from classifiers import TransparentMultinomialNB
from utils import ce_squared, load_imdb, ClassifierArchive
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from pickle import load, dump
import numpy as np


# In[2]:

t0 = time()

vect = CountVectorizer(min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1)) 

X_train, y_train, X_test, y_test, train_corpus, test_corpus = load_imdb("./aclImdb", shuffle=True, vectorizer=vect)

feature_names = vect.get_feature_names()
y_test_na = y_test[:, np.newaxis]
y_test_na = np.append(y_test_na, 1-y_test_na, axis=1)

duration = time() - t0

print("Loading took {:0.2f}s.\n".format(duration))


# # Experiment

# In[3]:

with open('clf.arch', 'rb') as f:
    clf_arch = load(f)

best_clf = clf_arch.classifiers[-1]
train_indices_set = set(clf_arch.train_indices[-1])
train_indices = clf_arch.train_indices[-1]
y_modified = np.copy(clf_arch.modified_labels[-1])
round_tag = clf_arch.round_tags[-1] + 1

current_error = ce_squared(y_test_na, best_clf.predict_proba(X_test))

for i in range(25000):
	
	if i in train_indices_set:
		# consider flipping or removing
		y_modified[i] = 1 - y_modified[i]
		clf_flipped = Transparen...
		clf_flipped.fit...
		flipped_error = 

		remove_train_indices = list(train_indices)
		remove_train_indices.remove(i)
		clf_remove = Tran
		clf_remove.fit(X_train[remove_train_indices],
		remove_error = 

		if flipped_error < current_error and flipped_error < remove_error:
			best_clf = clf_flipped
			current_error = flipped_error
		elif remove_error < current_error and remove_error <= flipped_error:
			best_clf = clf_remove
			train_indices = remove_train_indices
		else:
			y_modified[i] = 1 - y_modified[i]
	else:
		# consider adding with either 0 or 1
		add_train_indices = list(train_indices)
		add_train_indices.append(i)


		

for i in range(25000):

    added_now = False

    if i not in train_indices_set:
	added_now = True
        train_indices.append(i)
        train_indices.sort()
    
    clf = TransparentMultinomialNB()
    clf.fit(X_train[train_indices], y_modified[train_indices])        
    y_error = ce_squared(y_test_na, clf.predict_proba(X_test))
    
    y_modified[i] = 1 - y_modified[i]
    clf0 = TransparentMultinomialNB()
    clf0.fit(X_train[train_indices], y_modified[train_indices])  
    y0_error = ce_squared(y_test_na, clf.predict_proba(X_test))

    if y_error < current_error and y_error < y0_error:            
        current_error = y_error
        y_modified[i] = 1 - y_modified[i]
        best_clf = clf
        print("i = {}\tnew error = {:0.5f}".format(i, y_error))
    
    elif y0_error < current_error and y0_error < y_error: # switch back the label
        current_error = y0_error
        best_clf = clf0
        print("i = {}\tnew error = {:0.5f}".format(i, y0_error))
    
    elif added_now:
        train_indices.remove(i)

clf_arch.add_classifier(best_clf, train_indices, y_modified, round_tag)

with open('clf.arch', 'wb') as f:
    dump(clf_arch, f)
