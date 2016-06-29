
# coding: utf-8

# In[1]:

from classifiers import TransparentMultinomialNB
from utils import ce_squared, load_imdb, ClassifierArchive
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from time import time
from pickle import load, dump
import numpy as np


# In[2]:

t0 = time()

vect = TfidfVectorizer(min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1)) 

X_train, y_train, X_test, y_test, train_corpus, test_corpus = load_imdb("./aclImdb", shuffle=True, vectorizer=vect)

feature_names = vect.get_feature_names()

X_val = csr_matrix(X_train[:12500])
y_val = np.copy(y_train[:12500])

X_train = csr_matrix(X_train[12500:])
y_train = np.copy(X_train[12500:])

y_val_na = y_val[:, np.newaxis]
y_val_na = np.append(y_val_na, 1-y_val_na, axis=1)

duration = time() - t0

print("Loading took {:0.2f}s.\n".format(duration))


# # Experiment

# In[3]:

with open('clf4.arch', 'rb') as f:
    clf_arch = load(f)

best_clf = clf_arch.classifiers[-1]
train_indices_set = set(clf_arch.train_indices[-1])
train_indices = clf_arch.train_indices[-1]
y_modified = clf_arch.modified_labels[-1]
round_tag = clf_arch.round_tags[-1] + 1

current_error = ce_squared(y_val_na, best_clf.predict_proba(X_val))

for i in range(X_train.shape[0]):
    
    if i in train_indices_set:

        # consider flipping or removing
        y_modified[i] = 1 - y_modified[i]
        clf_flipped = TransparentMultinomialNB()
        clf_flipped.fit(X_train[train_indices], y_modified[train_indices])  
        flipped_error = ce_squared(y_val_na, clf_flipped.predict_proba(X_val))

        remove_train_indices = list(train_indices)
        remove_train_indices.remove(i)
        clf_remove = TransparentMultinomialNB()
        clf_remove.fit(X_train[remove_train_indices], y_modified[remove_train_indices])  
        remove_error = ce_squared(y_val_na, clf_remove.predict_proba(X_val))

        if flipped_error < current_error and flipped_error < remove_error:
            best_clf = clf_flipped
            current_error = flipped_error
            print("i = {}\tnew error = {:0.5f}".format(i, current_error))

        elif remove_error < current_error:
            best_clf = clf_remove
            train_indices = remove_train_indices
            current_error = remove_error
            print("i = {}\tnew error = {:0.5f}".format(i, current_error))

        else:
            y_modified[i] = 1 - y_modified[i]

    else:

        # consider adding with either 0 or 1
        train_indices.append(i)

        clf = TransparentMultinomialNB()
        clf.fit(X_train[train_indices], y_modified[train_indices])        
        y_error = ce_squared(y_val_na, clf.predict_proba(X_val))
        
        y_modified[i] = 1 - y_modified[i]
        clf0 = TransparentMultinomialNB()
        clf0.fit(X_train[train_indices], y_modified[train_indices])  
        y0_error = ce_squared(y_val_na, clf0.predict_proba(X_val))

        if y_error < current_error and y_error <= y0_error:   
            current_error = y_error
            y_modified[i] = 1 - y_modified[i]
            best_clf = clf
            print("i = {}\tnew error = {:0.5f}".format(i, current_error))
        
        elif y0_error < current_error:
            current_error = y0_error
            best_clf = clf0
            print("i = {}\tnew error = {:0.5f}".format(i, current_error))

        else:
            train_indices.pop()

with open('clf4.arch', 'rb') as f:
    clf_arch = load(f)

clf_arch.add_classifier(best_clf, train_indices, y_modified, round_tag)

with open('clf4.arch', 'wb') as f:
    dump(clf_arch, f)
