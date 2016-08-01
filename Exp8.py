from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
from concurrent.futures import ProcessPoolExecutor
from classifiers import TransparentMultinomialNB as Classifier
from utils import ce_squared, load_imdb, ClassifierArchive
from itertools import starmap, repeat, chain
from time import time
import numpy as np
import pickle

# Functions
def produce_modifications(X_train, y_train, train_indices, target_indices, X_val, y_val_na):
    for i in target_indices:

        if i in train_indices:
            mod0 = np.copy(y_train)
            mod0[i] = 1 - mod0[i]
            yield X_train, mod0, train_indices, X_val, y_val_na

            mod1 = list(train_indices)
            mod1.remove(i)
            yield X_train, y_train, mod1, X_val, y_val_na

        else:
            mod0 = list(train_indices)
            mod0.append(i)
            yield X_train, y_train, mod0, X_val, y_val_na

            mod1 = np.copy(y_train)
            mod1[i] = 1 - mod1[i]
            yield X_train, mod1, mod0, X_val, y_val_na

def test_modification(test):
    X_train, y_train, train_indices, X_val, y_val_na = test
    
    clf = Classifier()
    clf.fit(X_train[train_indices],y_train[train_indices])
    new_error = ce_squared(y_val_na, clf.predict_proba(X_val))
    
    return new_error, y_train, train_indices

# Loading
t0 = time()

vect = Vectorizer(min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1))

X_train, y_train, X_test, y_test, train_corpus, test_corpus = load_imdb("./aclImdb", shuffle=True, vectorizer=vect)

y_test_na = y_test[:, np.newaxis]
y_test_na = np.append(y_test_na, 1-y_test_na, axis=1)

clf = Classifier()
clf.fit(X_train, y_train)
ctrl_clf = clf
ctrl_error = ce_squared(y_test_na, clf.predict_proba(X_test))
ctrl_acc = clf.score(X_test, y_test)

# Split the train dataset in 2 for validation
split = int(X_train.shape[0] / 2)

X_val = csr_matrix(X_train[split:])
y_val = np.copy(y_train[split:])

X_train = csr_matrix(X_train[:split])
y_train = np.copy(y_train[:split])

y_val_na = y_val[:, np.newaxis]
y_val_na = np.append(y_val_na, 1-y_val_na, axis=1)

duration = time() - t0
print("Loading the dataset took {:0.2f}s.".format(duration), '\n')

# Experiment
start_ind = 0
batch_size = 10
end_ind = start_ind + batch_size

clf = Classifier()
clf.fit(X_train, y_train)
best_error = ce_squared(y_val_na, clf.predict_proba(X_val))
best_y_train = np.copy(y_train)
best_train_indices = list(range(X_train.shape[0]))
with ProcessPoolExecutor() as executor:
    while end_ind <= X_train.shape[0]:
        target_indices = range(start_ind, end_ind)
        mods = produce_modifications(X_train, best_y_train, best_train_indices, target_indices, X_val, y_val_na)
        test_results = list(executor.map(test_modification, mods))
        test_results.append((best_error, best_y_train, best_train_indices))
        best_error, best_y_train, best_train_indices = min(test_results, key=lambda x: x[0])
        print('Training round: 0,\tProcessed: {:5d} samples,\tcurrent error is {:0.4f}'.format(end_ind, best_error))
        start_ind += batch_size
        end_ind += batch_size
        
    best_clf = Classifier()
    best_clf.fit(X_train[best_train_indices], best_y_train[best_train_indices])
    test_acc = best_clf.score(X_test, y_test)
    print('Training round: 0,\tTest accuracy is {:0.3f},\tCotrol accuracy is {:0.3f}'.format(test_acc, ctrl_acc))

    clf_arch = ClassifierArchive(ctrl_clf, best_clf, best_train_indices, best_y_train, vect)

    with open('clf8.arch', 'wb') as f:
        pickle.dump(clf_arch, f)
        
    for i in range(2, 11):
        start_ind = 0
        end_ind = start_ind + batch_size
        
        while end_ind <= X_train.shape[0]:
            target_indices = range(start_ind, end_ind)
            mods = produce_modifications(X_train, best_y_train, best_train_indices, target_indices, X_val, y_val_na)
            test_results = list(executor.map(test_modification, mods))
            test_results.append((best_error, best_y_train, best_train_indices))
            best_error, best_y_train, best_train_indices = min(test_results, key=lambda x: x[0])
            
            print('Training round: {},\tProcessed: {:5d} samples,\tcurrent error is {:0.4f}'.format(i, end_ind, best_error))
            start_ind += batch_size
            end_ind += batch_size
        
        best_clf = Classifier()
        best_clf.fit(X_train[best_train_indices], best_y_train[best_train_indices])
        test_acc = best_clf.score(X_test, y_test)
        print('Training round: {},\tTest accuracy is {:0.3f},\tCotrol accuracy is {:0.3f}'.format(i, test_acc, ctrl_acc))

        with open('clf8.arch', 'rb') as f:
            clf_arch = pickle.load(f)
            
        clf_arch.add_classifier(best_clf, best_train_indices, best_y_train, i)

        with open('clf8.arch', 'wb') as f:
            pickle.dump(clf_arch, f)
        
    print('Experiment is done.')
