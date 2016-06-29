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

X_train, y_train, X_test, y_test, train_corpus, test_corpus = load_imdb("./aclImdb", shuffle=True, vectorizer=vect)

feature_names = vect.get_feature_names()

clf = TransparentMultinomialNB()
clf.fit(X_train, y_train)
best_clf = clf
ctrl_clf = clf

y_test_na = y_test[:, np.newaxis]
y_test_na = np.append(y_test_na, 1-y_test_na, axis=1)
current_error= ce_squared(y_test_na, clf.predict_proba(X_test))

X_val = csr_matrix(X_train[12500:])
y_val = np.copy(y_train[12500:])

X_train = csr_matrix(X_train[:12500])
y_train = np.copy(y_train[:12500])

y_val_na = y_val[:, np.newaxis]
y_val_na = np.append(y_val_na, 1-y_val_na, axis=1)

y_modified = np.copy(y_train)

duration = time() - t0

print("Loading took {:0.2f}s.\n".format(duration))

# In[6]:


# In[7]:
train_indices = list(range(X_train.shape[0]))

for i in range(X_train.shape[0]):

    # Flip labels
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
        print('i = {}\tnew_error = {:0.5f} flipped'.format(i, current_error))

    elif remove_error < current_error:
        best_clf = clf_remove
        current_error = remove_error
        train_indices = remove_train_indices
        print('i = {}\tnew_error = {:0.5f} removed'.format(i, current_error))

    else:
        y_modified[i] = 1 - y_modified[i]

arch = ClassifierArchive(ctrl_clf, best_clf, train_indices, y_modified, vect)

with open('clf4.arch', 'wb') as f:
    dump(arch, f)
