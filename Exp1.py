
# coding: utf-8

# In[1]:

from classifiers import TransparentMultinomialNB
from utils import ce_squared, load_imdb, ColoredWeightedDoc, TopInstances
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
        train_indices.pop()


# In[4]:

ctrl_clf = TransparentMultinomialNB()
ctrl_clf.fit(X_train, y_train)


# In[5]:

x = ctrl_clf.predict_proba(X_test) - best_clf.predict_proba(X_test)
x = np.absolute(x[:,0])
i = np.argsort(x)[0]
with open('best.clf', 'wb') as f:
    dump(best_clf,f)

with open('ctrl.clf', 'wb') as f:
    dump(ctrl_clf, f)
# In[6]:

#neg_evi, pos_evi = best_clf.predict_evidences(X_test)
#i = TopInstances(neg_evi, pos_evi, best_clf.get_bias()).most_negatives()[0]
display_html("<b>"+'Best Classifier'+"<b>", raw=True)
display(ColoredWeightedDoc(test_corpus[i], feature_names, best_clf.get_weights()))
display_html("<b>"+'Control Classifier'+"<b>", raw=True)
display(ColoredWeightedDoc(test_corpus[i], feature_names, ctrl_clf.get_weights()))

