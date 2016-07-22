import numpy as np
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
from scipy.sparse import csr_matrix
from time import time
from classifiers import TransparentMultinomialNB as Classifier
from mutils import improve_labels
from utils import load_imdb, ce_squared, ClassifierArchive
from pickle import dump


# Load the dataset
t0 = time()

vect = Vectorizer(min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1))

X_train, y_train, X_test, y_test, train_corpus, test_corpus = load_imdb("./aclImdb", shuffle=True, vectorizer=vect)
y_test_na = y_test[:, np.newaxis]
y_test_na = np.append(y_test_na, 1-y_test_na, axis=1)

clf = Classifier()
clf.fit(X_train, y_train)
best_clf = clf
ctrl_clf = clf
current_error= ce_squared(y_test_na, clf.predict_proba(X_test))

split = X_train.shape[0] / 2

X_val = csr_matrix(X_train[split:])
y_val = np.copy(y_train[split:])

X_train = csr_matrix(X_train[:split])
y_train = np.copy(y_train[:split])

y_val_na = y_val[:, np.newaxis]
y_val_na = np.append(y_val_na, 1-y_val_na, axis=1)

duration = time() - t0
print("Loading the dataset took {:0.2f}s.".format(duration), '\n')


# Start the first round of training
train_indices = list(range(X_train.shape[0]))

for i in range(0, X_train.shape[0], 10):
    target_indices = list(range(i, i+9))
    train_indices, y_train = improve_labels(X_train, y_train, X_val, y_val, train_indices, target_indices)


# Store the trained classifier
arch = ClassifierArchive(ctrl_clf, best_clf, train_indices, y_modified, vect)

with open('clf6.arch', 'wb') as f:
    dump(arch, f)
