import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from time import time
from classifiers import TransparentMultinomialNB
from utils import load_imdb, ce_squared, ClassifierArchive
from pickle import dump


# In[3]:

t0 = time()

vect = TfidfVectorizer(min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1))

X_train, y_train, X_val, y_val, X_test, y_test, *_ = load_imdb("./aclImdb", shuffle=True, random_state=11, vectorizer=vect)

feature_names = vect.get_feature_names()

clf = TransparentMultinomialNB()
clf.fit(X_train, y_train)
ctrl_clf = clf

y_test_na = y_test[:, np.newaxis]
y_test_na = np.append(y_test_na, 1-y_test_na, axis=1)

y_val_na = y_val[:, np.newaxis]
y_val_na = np.append(y_val_na, 1-y_val_na, axis=1)

y_modified = np.copy(y_train)

duration = time() - t0

print("Loading took {:0.2f}s.\n".format(duration))

# In[6]:


# In[7]:
train_indices = list(range(10))
best_clf = TransparentMultinomialNB()
best_clf.fit(X_train[train_indices], y_train[train_indices])
current_error = ce_squared(y_val_na, best_clf.predict_proba(X_val))

for i in range(10, X_train.shape[0]):

    train_indices.append(i)
    clf = TransparentMultinomialNB()
    clf.fit(X_train[train_indices], y_modified[train_indices])
    add_error = ce_squared(y_val_na, clf.predict_proba(X_val))

    # Flip labels
    y_modified[i] = 1 - y_modified[i]
    clf_flipped = TransparentMultinomialNB()
    clf_flipped.fit(X_train[train_indices], y_modified[train_indices])
    flipped_error = ce_squared(y_val_na, clf_flipped.predict_proba(X_val))

    if flipped_error < current_error and flipped_error < add_error:
        best_clf = clf_flipped
        current_error = flipped_error
        print('i = {}\tnew_error = {:0.5f} flipped'.format(i, current_error))

    elif add_error < current_error:
        best_clf = clf
        current_error = add_error
        y_modified[i] = 1 - y_modified[i]
        print('i = {}\tnew_error = {:0.5f} added'.format(i, current_error))

    else:
        train_indices.pop()

print(best_clf.score(X_test, y_test))
arch = ClassifierArchive(ctrl_clf, best_clf, train_indices, y_modified, vect)

with open('clf11.arch', 'wb') as f:
    dump(arch, f)
