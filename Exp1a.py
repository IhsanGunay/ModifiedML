
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

    if i not in train_indices_set:
        train_indices.append(i)
    
    clf = TransparentMultinomialNB()
    clf.fit(X_train[train_indices], y_modified[train_indices])        
    y_error = ce_squared(y_test_na, clf.predict_proba(X_test))
    
    y_modified[i] = 1 - y_modified[i]
    clf = TransparentMultinomialNB()
    clf.fit(X_train[train_indices], y_modified[train_indices])  
    y0_error = ce_squared(y_test_na, clf.predict_proba(X_test))

    if y_error < current_error and y_error < y0_error:            
        current_error = y_error
        y_modified[i] = 1 - y_modified[i]
        clf = TransparentMultinomialNB()
        clf.fit(X_train[train_indices], y_modified[train_indices]) 
        best_clf = clf
        print("i = {}\tnew error = {:0.5f}".format(i, y_error))
    
    elif y0_error < current_error and y0_error < y_error: # switch back the label
        current_error = y0_error
        best_clf = clf
        print("i = {}\tnew error = {:0.5f}".format(i, y0_error))
    
    else:
        train_indices.remove(i)

clf_arch.add_classifier(best_clf, train_indices, y_modified, round_tag)

with open('clf.arch', 'wb') as f:
    dump(clf_arch, f)
