from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
from sklearn.metrics import classification_report, confusion_matrix
from classifiers import TransparentMultinomialNB as Classifier
from utils import ce_squared, load_imdb, ClassifierArchive
from itertools import starmap, repeat, chain
from time import time
import numpy as np
import pickle
import sys

# Functions
def validate_modification(X_train, y_train, train_indices, validation_list, best_error):
    
    tests = []
    for X_val, y_val_na in validation_list:
        test = X_train, y_train, train_indices, X_val, y_val_na
        tests.append(test)
        
    results = map(test_modification, tests)
    improvements = 0  
    error_sum = 0
    
    for val_error in results:
        if val_error <= best_error:
            improvements += 1
            error_sum += val_error
    if improvements > N/2:
        avg_error = error_sum / improvements    
        if avg_error < best_error:
            return True, avg_error
        else:
            return True, best_error
    else:
        return False, best_error
    
def test_modification(test):
    X_train, y_train, train_indices, X_val, y_val_na = test
    
    clf = Classifier()
    clf.fit(X_train[train_indices],y_train[train_indices])
    new_error = ce_squared(y_val_na, clf.predict_proba(X_val))
    
    return new_error

# Loading
t0 = time()

vect = Vectorizer(min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1))

X_train, y_train, X_val, y_val, X_test, y_test, *_ = load_imdb("./aclImdb", shuffle=True, vectorizer=vect)

y_test_na = y_test[:, np.newaxis]
y_test_na = np.append(y_test_na, 1-y_test_na, axis=1)

y_val_na = y_val[:, np.newaxis]
y_val_na = np.append(y_val_na, 1-y_val_na, axis=1)

N = int(sys.argv[1])
print('Number of validation sets is {}'.format(N))
train_indices = np.arange(X_train.shape[0])
val_indices = np.array_split(train_indices, N)

validation_list = []
for inds in val_indices:
    y_valn = np.copy(y_val_na[inds])
    X_valn = csr_matrix(X_val[inds])
    val_set = X_valn, y_valn
    validation_list.append(val_set)
    
clf = Classifier()
clf.fit(X_train, y_train)
ctrl_clf = clf
ctrl_error = ce_squared(y_test_na, clf.predict_proba(X_test))
ctrl_acc = clf.score(X_test, y_test)

duration = time() - t0
print("Loading the dataset took {:0.2f}s.".format(duration), '\n')

# Experiment
clf = Classifier()
clf.fit(X_train, y_train)
best_error = ce_squared(y_val_na, clf.predict_proba(X_val))
y_modified = np.copy(y_train)
train_indices = list(train_indices)

for i in range(X_train.shape[0]):
    if i in train_indices:
        y_modified[i] = 1 - y_modified[i]
        is_validated_0, try_error_0 = validate_modification(X_train, y_modified, train_indices, validation_list, best_error)

        remove_train_indices = list(train_indices)
        remove_train_indices.remove(i)
        is_validated_1, try_error_1 =  validate_modification(X_train, y_modified, train_indices, validation_list, best_error)

        if is_validated_0:           
            if is_validated_1:
                if try_error_0 < try_error_1:
                    best_error = try_error_0
                    print('Round: {:5d}, error = {:0.4f}, code: 21'.format(i, best_error))

                else:
                    best_error = try_error_1
                    train_indices = remove_train_indices
                    print('Round: {:5d}, error = {:0.4f}, code: 12'.format(i, best_error))

            else:
                best_error = try_error_0
                print('Round: {:5d}, error = {:0.4f}, code: 20'.format(i, best_error))

        elif is_validated_1:
            best_error = try_error_1
            train_indices = remove_train_indices
            print('Round: {:5d}, error = {:0.4f}, code: 02'.format(i, best_error))

        else:
            y_modified[i] = 1 - y_modified[i]

    else:
        train_indices.append(i)
        is_validated_0, try_error_0 = validate_modification(X_train, y_modified, train_indices, validation_list, best_error)

        y_modified[i] = 1 - y_modified[i]
        is_validated_1, try_error_1 = validate_modification(X_train, y_modified, train_indices, validation_list, best_error)

        if is_validated_0:           
            if is_validated_1:
                if try_error_0 < try_error_1:
                    best_error = try_error_0
                    print('Round: {:5d}, error = {:0.4f}, code: 21'.format(i, best_error))

                else:
                    best_error = try_error_1
                    print('Round: {:5d}, error = {:0.4f}, code: 12'.format(i, best_error))

            else:
                best_error = try_error_0
                print('Round: {:5d}, error = {:0.4f}, code: 20'.format(i, best_error))

        elif is_validated_1:
            best_error = try_error_1
            train_indices = remove_train_indices
            print('Round: {:5d}, error = {:0.4f}, code: 02'.format(i, best_error))

        else:
            train_indices.pop()
    
    best_clf = Classifier()
    best_clf.fit(X_train[train_indices], y_modified[train_indices])
test_acc = best_clf.score(X_test, y_test)
print('Training Round: 0,\tTest accuracy is {:0.3f},\tCotrol accuracy is {:0.3f}'.format(test_acc, ctrl_acc))

clf_arch = ClassifierArchive(ctrl_clf, best_clf, train_indices, y_modified, vect)

with open('clf9.arch', 'wb') as f:
    pickle.dump(clf_arch, f)
    
for round_tag in range(2, 11):
    for i in range(X_train.shape[0]):
        if i in train_indices:
            y_modified[i] = 1 - y_modified[i]
            is_validated_0, try_error_0 = validate_modification(X_train, y_modified, train_indices, validation_list, best_error)

            remove_train_indices = list(train_indices)
            remove_train_indices.remove(i)
            is_validated_1, try_error_1 = validate_modification(X_train, y_modified, train_indices, validation_list, best_error)

            if is_validated_0:           
                if is_validated_1:
                    if try_error_0 < try_error_1:
                        best_error = try_error_0
                        print('Round: {:5d}, error = {:0.4f}, code: 21'.format(i, best_error))

                    else:
                        best_error = try_error_1
                        train_indices = remove_train_indices
                        print('Round: {:5d}, error = {:0.4f}, code: 12'.format(i, best_error))

                else:
                    best_error = try_error_0
                    print('Round: {:5d}, error = {:0.4f}, code: 20'.format(i, best_error))

            elif is_validated_1:
                best_error = try_error_1
                train_indices = remove_train_indices
                print('Round: {:5d}, error = {:0.4f}, code: 02'.format(i, best_error))

            else:
                y_modified[i] = 1 - y_modified[i]

        else:
            train_indices.append(i)
            is_validated_0, try_error_0 = validate_modification(X_train, y_modified, train_indices, validation_list, best_error)

            y_modified[i] = 1 - y_modified[i]
            is_validated_1, try_error_1 = validate_modification(X_train, y_modified, train_indices, validation_list, best_error)

            if is_validated_0:           
                if is_validated_1:
                    if try_error_0 < try_error_1:
                        best_error = try_error_0
                        print('Round: {:5d}, error = {:0.4f}, code: 21'.format(i, best_error))

                    else:
                        best_error = try_error_1
                        print('Round: {:5d}, error = {:0.4f}, code: 12'.format(i, best_error))

                else:
                    best_error = try_error_0
                    print('Round: {:5d}, error = {:0.4f}, code: 20'.format(i, best_error))

            elif is_validated_1:
                best_error = try_error_1
                train_indices = remove_train_indices
                print('Round: {:5d}, error = {:0.4f}, code: 02'.format(i, best_error))

            else:
                train_indices.pop()
        
        best_clf = Classifier()
        best_clf.fit(X_train[train_indices], y_modified[train_indices])
    test_acc = best_clf.score(X_test, y_test)
    print('Training Round: {},\tTest accuracy is {:0.3f},\tCotrol accuracy is {:0.3f}'.format(round_tag, test_acc, ctrl_acc))

    with open('clf9.arch', 'rb') as f:
        clf_arch = pickle.load(f)
        
    clf_arch.add_classifier(best_clf, train_indices, y_modified, round_tag)

    with open('clf9.arch', 'wb') as f:
        pickle.dump(clf_arch, f)
        
print('Experiment is done.')

with open('clf9.arch', 'rb') as f:
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

test_acc = ce_squared(y_val_na, best_clf.predict_proba(X_val))
print('Best validation error is {}'.format(test_acc), '\n')

test_acc = ce_squared(y_test_na, best_clf.predict_proba(X_test))
print('Best test error is {}'.format(test_acc))

best_pred = best_clf.predict(X_test)
ctrl_pred = ctrl_clf.predict(X_test)
print('Best Classifier')
print('Control Classifier')
print(classification_report(y_test, ctrl_pred))
print(classification_report(y_test, best_pred))

best_weights = best_clf.get_weights()
ctrl_weights = ctrl_clf.get_weights()

best_ws = np.argsort(best_weights)
ctrl_ws  = np.argsort(ctrl_weights)

print("Top Positive")
print(" ".join(["{} ({})".format(feature_names[i], ctrl_clf.feature_count_[:,i])
                for i in ctrl_ws[-10:][::-1]]))

print("\nTop Negative")
print(" ".join(["{} ({})".format(feature_names[i], ctrl_clf.feature_count_[:,i])
                for i in ctrl_ws[:10]]))

print("Top Positive")
print(" ".join(["{} ({})".format(feature_names[i], best_clf.feature_count_[:,i])
                for i in best_ws[-10:][::-1]]))

print("\nTop Negative")
print(" ".join(["{} ({})".format(feature_names[i], best_clf.feature_count_[:,i])
                for i in best_ws[:10]]))

x = ctrl_clf.predict_proba(X_test) - best_clf.predict_proba(X_test)
x = np.absolute(x[:,0])
inds = np.argsort(x)
i = inds[-1]
print(ctrl_clf.predict_proba(X_test)[i]) 
print(best_clf.predict_proba(X_test)[i])
