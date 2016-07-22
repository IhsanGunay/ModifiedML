from sklearn.base import clone
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecuter
from utils import ce_squared
import numpy as np
import multiprocessing as mp

def produce_modifications(q, y_train, start_index, mod_length):
    for i in range(start_index, start_index + mod_length):
        mod0 = np.copy(y_train)
        mod0[i] = 1 - mod0[i]
        q.put(mod0)
        mod1 = np.delete(y_train, i)
        q.put(mod1)

def test_modification(q, coroutine, classifier, current_error, X_train, X_val, y_val_na, train_indices):
    clf = clone(classifier)
    while not q.empty():
        y_train = q.get()
        clf.fit(X_train[train_indices],y_train[train_indices])
        new_error = ce_squared(y_val_na, clf.predict_proba(X_val))
        if new_error < current_error:
            result = new_error, y_train, train_indices
        else:
            result = current_error, y_train, train_indices
        coroutine.send(result)


def improve_labels(classifier, current_error, X_train, y_train, X_val, y_val_na, train_indices, start_index, mod_length):
    best_error = current_error
    best_y_train = y_train
    best_train_indices = train_indices

    def select_best():
        nonlocal best_error
        nonlocal best_y_train
        nonlocal best_train_indices
        while True:
            try_error, try_y_train, try_train_indices = (yield)
            if try_error < min_error:
                best_error = try_error
                best_y_train = try_y_train
                best_train_indices = try_train_indices
    
    best_mod.next() # Prime the coroutine
    mods = mp.Queue()

    producer = mp.Process(target=produce_modifications, args=(mods, y_train, start_index, mod_length))
    producer.start()

    for _ in range(2):
        consumer = mp.Process(target=test_modification, args=(mods, select_best, classifier, current_error, ))
    return best_clf, y_modified, train_indices



def is_validated():



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
