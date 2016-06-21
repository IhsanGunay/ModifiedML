
# coding: utf-8

# In[1]:

from classifiers import TransparentMultinomialNB
from utils import ce_squared, load_imdb, ColoredWeightedDoc, ClassifierArchive
from sklearn.feature_extraction.text import CountVectorizer
from IPython.display import display, display_html
from time import time
from pickle import dump
import numpy as np


# In[2]:

t0 = time()

vect = CountVectorizer(min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1)) 

X_train, y_train, X_test, y_test, train_corpus, test_corpus = load_imdb("./aclImdb", shuffle=True, vectorizer=vect)

feature_names = vect.get_feature_names()
y_test_na = y_test[:, np.newaxis]
y_test_na = np.append(y_test_na, 1-y_test_na, axis=1)
y_modified = np.copy(y_train)


duration = time() - t0

print("Loading took {:0.2f}s.\n".format(duration))


# # Experiment

# In[3]:

clf = TransparentMultinomialNB()
train_indices = list(range(10))

clf.fit(X_train[train_indices], y_train[train_indices])

best_clf = clf
current_error = ce_squared(y_test_na, clf.predict_proba(X_test))

for i in range(10,25000):
    train_indices.append(i)
    
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
    
    else:
        train_indices.pop()

# In[4]:

ctrl_clf = TransparentMultinomialNB()
ctrl_clf.fit(X_train, y_train)

# In[5]:

arch = ClassifierArchive(ctrl_clf, best_clf, train_indices, y_modified, vect)

with open('clf.arch', 'wb') as f:
    dump(arch, f)
