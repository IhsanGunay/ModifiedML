'''
Created on Sep 16, 2015

@author: mbilgic
'''

import numpy as np
from glob import glob
from sklearn.feature_extraction.text import CountVectorizer
from time import time
import re

def load_imdb(path, split_half=True, shuffle=True, random_state=42, vectorizer=CountVectorizer(min_df=5, max_df=1.0, binary=True)):
    
    print("Loading the imdb reviews data")
    
    train_neg_files = glob(path + r"/train/neg/*.txt")
    train_pos_files = glob(path + r"/train/pos/*.txt")
    
    train_corpus = []
    
    y_train = []
    
    for tnf in train_neg_files:
        with open(tnf, 'r', errors='replace') as f:
            line = f.read()
            train_corpus.append(line)
            y_train.append(0)
            
    for tpf in train_pos_files:
        with open(tpf, 'r', errors='replace') as f:
            line = f.read()
            train_corpus.append(line)
            y_train.append(1)
            
    test_neg_files = glob(path + r"/test/neg/*.txt")
    test_pos_files = glob(path + r"/test/pos/*.txt")
    
    test_corpus = []
    
    y_test = []
    
    for tnf in test_neg_files:
        with open(tnf, 'r', errors='replace') as f:
            test_corpus.append(f.read())
            y_test.append(0)
            
    for tpf in test_pos_files:
        with open(tpf, 'r', errors='replace') as f:
            test_corpus.append(f.read())
            y_test.append(1)
                
    print("Data loaded.")
    
    print("Extracting features from the training dataset using a sparse vectorizer")
    print("Feature extraction technique is {}.".format(vectorizer))
    t0 = time()

    split = int(len(y_train)/2) 

    y_val = y_train[split:]
    y_train = y_train[:split]
    
    val_corpus = train_corpus[split:]
    train_corpus = train_corpus[:split]

    X_train = vectorizer.fit_transform(train_corpus)
    X_val = vectorizer.transform(val_corpus)
    
    duration = time() - t0
    print("done in {}s".format(duration))
    print(X_train.shape)
    print("n_samples: {}, n_features: {}".format(*X_train.shape), '\n')
        
    print("Extracting features from the test dataset using the same vectorizer")
    t0 = time()
        
    X_test = vectorizer.transform(test_corpus)
    
    duration = time() - t0
    print("done in {}s".format(duration))
    print("n_samples: {}, n_features: {}".format(*X_test.shape), '\n')
    
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(y_train))        
        
        X_train = X_train.tocsr()
        X_train = X_train[indices]
        y_train = y_train[indices]
        train_corpus_shuffled = [train_corpus[i] for i in indices]
        
        np.random.seed(random_state)
        indices = np.random.permutation(len(y_val))        
        print(indices)
        
        X_val = X_val.tocsr()
        X_val = X_val[indices]
        y_val = y_val[indices]
        val_corpus_shuffled = [val_corpus[i] for i in indices]
        
        np.random.seed(random_state)
        indices = np.random.permutation(len(y_test))
        
        X_test = X_test.tocsr()
        X_test = X_test[indices]
        y_test = y_test[indices]
        test_corpus_shuffled = [test_corpus[i] for i in indices]
         
    return X_train, y_train, X_val, y_val, X_test, y_test, train_corpus_shuffled, val_corpus_shuffled, test_corpus_shuffled

def ce_squared(T, probs):
    return ((T*probs)**2).sum()/len(probs)

class ColoredDoc(object):

    def __init__(self, doc, feature_names, coefs):
        self.doc = doc
        self.feature_names = feature_names
        self.coefs = coefs
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")

    def _repr_html_(self):
        html_rep = ""
        tokens = self.doc.split(" ")        
        for token in tokens:
            vocab_tokens = self.token_pattern.findall(token.lower())
            if len(vocab_tokens) > 0:
                vocab_token = vocab_tokens[0]
                try:
                    vocab_index = self.feature_names.index(vocab_token)
                    if self.coefs[vocab_index] > 0:
                        html_rep = html_rep + "<font color=blue> " + token + " </font>"
                    elif self.coefs[vocab_index] < 0:
                        html_rep = html_rep + "<font color=red> " + token + " </font>"
                    else:
                        html_rep = html_rep + "<font color=gray> " + token + " </font>"
                except:
                    html_rep = html_rep + "<font color=gray> " + token + " </font>"
            else:
                html_rep = html_rep + "<font color=gray> " + token + " </font>"
        return html_rep

class ColoredWeightedDoc(object):

    def __init__(self, doc, feature_names, coefs, binary = False):
        self.doc = doc
        self.feature_names = feature_names
        self.coefs = coefs
        self.binary = binary
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.abs_ranges = np.linspace(0, max([abs(coefs.min()), abs(coefs.max())]), 8)

    def _repr_html_(self):
        html_rep = ""
        tokens = self.doc.split(" ") 
        if self.binary:
            seen_tokens = set()       
        for token in tokens:
            vocab_tokens = self.token_pattern.findall(token.lower())
            if len(vocab_tokens) > 0:
                vocab_token = vocab_tokens[0]
                try:
                    vocab_index = self.feature_names.index(vocab_token)
                    
                    if not self.binary or vocab_index not in seen_tokens:
                        
                        if self.coefs[vocab_index] > 0: # positive word
                            for i in range(1, 7):
                                if self.coefs[vocab_index] < self.abs_ranges[i]:
                                    break
                            html_rep = html_rep + "<font size = " + str(i) + ", color=blue> " + token + " </font>"
                        
                        elif self.coefs[vocab_index] < 0: # negative word
                            for i in range(1, 7):
                                if self.coefs[vocab_index] > -self.abs_ranges[i]:
                                    break
                            html_rep = html_rep + "<font size = " + str(i) + ", color=red> " + token + " </font>"
                        
                        else: # neutral word
                            html_rep = html_rep + "<font size = 1, color=gray> " + token + " </font>"
                        
                        if self.binary:    
                            seen_tokens.add(vocab_index)
                    
                    else: # if binary and this is a token we have seen before
                        html_rep = html_rep + "<font size = 1, color=gray> " + token + " </font>"
                except: # this token does not exist in the vocabulary
                    html_rep = html_rep + "<font size = 1, color=gray> " + token + " </font>"
            else:
                html_rep = html_rep + "<font size = 1, color=gray> " + token + " </font>"
        return html_rep
    
class TopInstances():

    def __init__(self, neg_evis, pos_evis, intercept=0):
        self.neg_evis = neg_evis
        self.pos_evis = pos_evis
        self.intercept = intercept
        self.total_evis = self.neg_evis + self.pos_evis
        self.total_evis += self.intercept
        self.total_abs_evis = abs(self.neg_evis) + abs(self.pos_evis)
        self.total_abs_evis += abs(self.intercept)
        
    def most_negatives(self, k=1):
        evi_sorted = np.argsort(self.total_evis)
        return evi_sorted[:k]
    
    def most_positives(self, k=1):
        evi_sorted = np.argsort(self.total_evis)
        return evi_sorted[-k:][::-1]
    
    def least_opinionateds(self, k=1):
        abs_evi_sorted = np.argsort(self.total_abs_evis)
        return abs_evi_sorted[:k]
    
    def most_opinionateds(self, k=1):
        abs_evi_sorted = np.argsort(self.total_abs_evis)
        return abs_evi_sorted[-k:][::-1]
    
    def most_uncertains(self, k=1):
        abs_total_evis = abs(self.total_evis)
        abs_total_evi_sorted = np.argsort(abs_total_evis)
        return abs_total_evi_sorted[:k]
    
    def most_conflicteds(self, k=1):
        conflicts = np.min([abs(self.neg_evis), abs(self.pos_evis)], axis=0)
        conflict_sorted = np.argsort(conflicts)
        return conflict_sorted[-k:][::-1]
    
    def least_conflicteds(self, k=1):
        conflicts = np.min([abs(self.neg_evis), abs(self.pos_evis)], axis=0)
        conflict_sorted = np.argsort(conflicts)
        return conflict_sorted[:k]

class ClassifierArchive():

    def __init__(self, ctrl_clf, best_clf, train_indices, modified_labels, vect):
        self.vect = vect
        self.type = type(best_clf)
        self.ctrl_clf = ctrl_clf
        self.classifiers = [best_clf]
        self.train_indices = [train_indices]
        self.modified_labels = [modified_labels]
        self.round_tags = [1]
        assert type(best_clf) == type(ctrl_clf)

    def __len__(self):
        return len(self.classifiers)

    def stats(self):
        print(self.type, "\n")
        print(self.round_tags, "\n")
        print(self.vect)

    def add_classifier(self, clf, train_indices, modified_labels, round_tag):
        self.classifiers.append(clf)
        self.train_indices.append(train_indices)
        self.modified_labels.append(modified_labels)
        self.round_tags.append(round_tag)
        assert self.type == type(clf)
        assert len(self.classifiers) == len(self.train_indices)
        assert len(self.classifiers) == len(self.round_tags)
        assert len(self.classifiers) == len(self.modified_labels)

    def rm_classifier(round_tag):
        i = round_tags.index(round_tag)
        classifiers.pop(i)
        train_indices.pop(i)
        modified_labels.pop(i)
        round_tags.pop(i)
        assert len(self.classifiers) == len(self.train_indices)
        assert len(self.classifiers) == len(self.round_tags)
        assert len(self.classifiers) == len(self.modified_labels)
