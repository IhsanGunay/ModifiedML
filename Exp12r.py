
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
from sklearn.metrics import classification_report
from concurrent.futures import ProcessPoolExecutor
from classifiers import TransparentLogisticRegression as Classifier
from utils import ce_squared, load_imdb, ClassifierArchive
from time import time
import numpy as np
import pickle

t0 = time()

vect = Vectorizer(min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1))

X_train, y_train, X_val, y_val, X_test, y_test, train_corpus, val_corpus, test_corpus = load_imdb("./aclImdb", shuffle=True, vectorizer=vect)

y_test_na = y_test[:, np.newaxis]
y_test_na = np.append(y_test_na, 1-y_test_na, axis=1)

y_val_na = y_val[:, np.newaxis]
y_val_na = np.append(y_val_na, 1-y_val_na, axis=1)

clf = Classifier()
clf.fit(X_train, y_train)
ctrl_clf = clf
ctrl_error = ce_squared(y_test_na, clf.predict_proba(X_test))
ctrl_acc = clf.score(X_test, y_test)

duration = time() - t0
print("Loading the dataset took {:0.2f}s.".format(duration), '\n')

with open('clf12.arch', 'rb') as f:
    clf_arch = pickle.load(f)

clf_arch.stats()

ctrl_clf = clf_arch.ctrl_clf
best_clf = clf_arch.classifiers[-1]
first_clf = clf_arch.classifiers[0]
train_indices = clf_arch.train_indices[-1]
y_modified = np.copy(clf_arch.modified_labels[-1])

print('Number of samples used :', len(train_indices))
changed_labels = np.array(list(filter(lambda x: x[0]!=x[1], zip(y_modified[train_indices], y_train[train_indices]))))
print('Number of labels modified:', len(changed_labels))

changes = changed_labels[:,0] - changed_labels[:,1]
print('1 to 0 :', len(list(filter(lambda x: x<0, changes))))
print('0 to 1 :', len(list(filter(lambda x: x>0, changes))))

test_acc = ctrl_clf.score(X_test, y_test)
print('Control test accuracy is {}'.format(test_acc), '\n')

best_pred = best_clf.predict(X_test)
ctrl_pred = ctrl_clf.predict(X_test)
print(classification_report(y_test, ctrl_pred))
print(classification_report(y_test, best_pred))

clf = Classifier()
clf.fit(X_train, y_train)

val_acc = clf.score(X_val, y_val)
print('Initial validation accuracy is {}'.format(val_acc), '\n')

test_acc = clf.score(X_test, y_test)
print('Initial test accuracy is {}'.format(test_acc), '\n')

val_acc = first_clf.score(X_val, y_val)
print('First validation accuracy is {}'.format(val_acc), '\n')

test_acc = first_clf.score(X_test, y_test)
print('First test accuracy is {}'.format(test_acc), '\n')

val_acc = best_clf.score(X_val, y_val)
print('Best validation accuracy is {}'.format(val_acc), '\n')

test_acc = best_clf.score(X_test, y_test)
print('Best test accuracy is {}'.format(test_acc), '\n')

test_acc = ce_squared(y_val_na, first_clf.predict_proba(X_val))
print('First validation error is {}'.format(test_acc), '\n')
