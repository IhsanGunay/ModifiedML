from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
from concurrent.futures import ProcessPoolExecutor
from classifiers import TransparentLogisticRegression as Classifier
from utils import ce_squared, load_imdb, ClassifierArchive
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
        print('Training round: 1,\tProcessed: {:5d} samples,\tcurrent error is {:0.4f}'.format(end_ind, best_error))
        start_ind += batch_size
        end_ind += batch_size
        
    best_clf = Classifier()
    best_clf.fit(X_train[best_train_indices], best_y_train[best_train_indices])
    test_acc = best_clf.score(X_test, y_test)
    print('Training round: 1,\tTest accuracy is {:0.3f},\tCotrol accuracy is {:0.3f}'.format(test_acc, ctrl_acc))

    clf_arch = ClassifierArchive(ctrl_clf, best_clf, best_train_indices, best_y_train, vect)

    with open('clf12.arch', 'wb') as f:
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

        with open('clf12.arch', 'rb') as f:
            clf_arch = pickle.load(f)
            
        clf_arch.add_classifier(best_clf, best_train_indices, best_y_train, i)

        with open('clf12.arch', 'wb') as f:
            pickle.dump(clf_arch, f)
        
    print('Experiment is done.')

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

test_acc = ce_squared(y_test_na, first_clf.predict_proba(X_test))
print('First test error is {}'.format(test_acc), '\n')

val_acc = ce_squared(y_val_na, best_clf.predict_proba(X_val))
print('Best validation error is {}'.format(val_acc), '\n')
