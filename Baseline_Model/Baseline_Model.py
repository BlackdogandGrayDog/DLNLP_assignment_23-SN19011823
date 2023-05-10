#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 05:00:30 2023

@author: ericwei
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import gensim.downloader as api
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors, FastText
import pandas as pd
import sys
sys.path.append('../Data_Preprocessing')
import Data_Preprocessing as dp
from sklearn.model_selection import train_test_split


def naive_bayes_classifier(X_train, X_val, y_train, y_val):
    """
    Train a Naive Bayes classifier on the given data and validate it.
    """
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    return accuracy, auc, precision, recall, f1


def logistic_regression_classifier(X_train, X_val, y_train, y_val):
    """
    Train and test a Logistic Regression classifier on the given data.
    """
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, logistic_regression.predict_proba(X_val)[:, 1])
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    return accuracy, auc, precision, recall, f1


def svm_classifier(X_train, X_val, y_train, y_val):
    """
    Train and test a Support Vector Machine classifier on the given data.
    """
    svm_model = SVC(probability=True)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, svm_model.predict_proba(X_val)[:, 1])
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    return accuracy, auc, precision, recall, f1


def random_forest_classifier(X_train, X_val, y_train, y_val):
    """
    Train and test a Random Forest classifier on the given data.
    """
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, random_forest.predict_proba(X_val)[:, 1])
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    return accuracy, auc, precision, recall, f1


def count_vector_embedding(train_data, validation_data):
    """
    Transform the data using CountVectorizer.
    """
    X_train, X_val, y_train, y_val = train_data['comment_text'], validation_data['comment_text'], train_data['toxicity'], validation_data['toxic']
    vectorizer = CountVectorizer()
    X_train_count = vectorizer.fit_transform(X_train)
    X_val_count = vectorizer.transform(X_val)
    return X_train_count, X_val_count, y_train, y_val


def tfidf_embedding(train_data, validation_data):
    """
    Transform the data using TfidfVectorizer.
    """
    X_train, X_val, y_train, y_val = train_data['comment_text'], validation_data['comment_text'], train_data['toxicity'], validation_data['toxic']
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    return X_train_tfidf, X_val_tfidf, y_train, y_val


def document_embeddings(tokens, embedding_model):
    """
    Helper function for document embeddings.
    
    Arguments:
    tokens -- a list of words in a document
    embedding_model -- pre-trained model

    Returns:
    A vector that represents the entire document.
    """
    embeddings = [embedding_model[word] for word in tokens if word in embedding_model]
    
    if not embeddings:
        # If no words in the document are in the model, return a zero vector
        return np.zeros(embedding_model.vector_size)
        
    # Take the mean of the word embeddings to get the document embedding
    return np.mean(embeddings, axis=0)


def glove_embedding(train_data, validation_data):
    """
    Transform the data using GloVe model.
    """
    nltk.download('punkt')
    train_data['tokens'] = train_data['comment_text'].apply(word_tokenize)
    validation_data['tokens'] = validation_data['comment_text'].apply(word_tokenize)
    
    glove_input_file = 'glove.6B.300d.txt'
    word2vec_output_file = 'glove.6B.300d.word2vec.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)
    glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    
    X_train = np.array(train_data['tokens'].apply(lambda x: document_embeddings(x, glove_model)).tolist())
    X_val = np.array(validation_data['tokens'].apply(lambda x: document_embeddings(x, glove_model)).tolist())
    y_train = train_data['toxicity']
    y_val = validation_data['toxic']

    return X_train, X_val, y_train, y_val


def fasttext_embedding(train_data, validation_data):
    """
    Transform the data using FastText model.
    """
    nltk.download('punkt')
    train_data['tokens'] = train_data['comment_text'].apply(word_tokenize)
    validation_data['tokens'] = validation_data['comment_text'].apply(word_tokenize)
    
    sentences = [word_tokenize(sentence) for sentence in pd.concat([train_data['comment_text'], validation_data['comment_text']])]
    model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    X_train = np.array(train_data['tokens'].apply(lambda x: document_embeddings(x, model.wv)).tolist())
    X_val = np.array(validation_data['tokens'].apply(lambda x: document_embeddings(x, model.wv)).tolist())
    y_train = train_data['toxicity']
    y_val = validation_data['toxic']

    return X_train, X_val, y_train, y_val


def word2vec_embedding(train_data, validation_data):
    """
    Transform the data using Word2Vec model.
    """
    nltk.download('punkt')
    train_data['tokens'] = train_data['comment_text'].apply(word_tokenize)
    validation_data['tokens'] = validation_data['comment_text'].apply(word_tokenize)
    
    word2vec_model = api.load('word2vec-google-news-300')
    
    X_train = np.array(train_data['tokens'].apply(lambda x: document_embeddings(x, word2vec_model)).tolist())
    X_val = np.array(validation_data['tokens'].apply(lambda x: document_embeddings(x, word2vec_model)).tolist())
    y_train = train_data['toxicity']
    y_val = validation_data['toxic']

    return X_train, X_val, y_train, y_val




def evaluate_classifiers(X_train, X_val, y_train, y_val, exclude_naive_bayes=False):
    """
    Evaluate Naive Bayes, Logistic Regression, SVM, and Random Forest classifiers on the given train and validation data.
    
    Arguments:
       X_train, X_val, y_train, y_val -- feature matrices and target vectors for train and validation data
       exclude_naive_bayes -- flag to exclude Naive Bayes classifier due to negative values in input data

    Returns:
    A dictionary where keys are classifier names and values are their corresponding metrics.
    """
    classifiers = {
        'Logistic Regression': logistic_regression_classifier,
        'SVM': svm_classifier,
        'Random Forest': random_forest_classifier,
    }

    if not exclude_naive_bayes:
        classifiers['Naive Bayes'] = naive_bayes_classifier

    metrics = {}

    for classifier_name, classifier_function in classifiers.items():
        accuracy, auc, precision, recall, f1 = classifier_function(X_train, X_val, y_train, y_val)
        metrics[classifier_name] = {
            'accuracy': accuracy,
            'AUC': auc,
            'precision': precision,
            'recall': recall,
            'F1-score': f1
        }

    return metrics


def run_evaluation(Dataset_path, val = True):
    """
    Evaluate classifiers on different embeddings given a path to the dataset.
    
    Arguments:
    Dataset_path -- the path to the dataset containing train, validation, and test data.
    """
    # Load train and validation data
    train_data, validation_data, _ = dp.preprocess_datasets(Dataset_path, lang = False)
    if val:
        print('\nMonolingual...')
        train_data, validation_data = train_test_split(train_data, test_size=0.2, random_state=42)
    else:
        print('\nMultilingual...')

    # Transform the data using different embeddings
    X_train_count, X_val_count, y_train_count, y_val_count = count_vector_embedding(train_data, validation_data)
    X_train_tfidf, X_val_tfidf, y_train_tfidf, y_val_tfidf = tfidf_embedding(train_data, validation_data)
    X_train_glove, X_val_glove, y_train_glove, y_val_glove = glove_embedding(train_data, validation_data)
    X_train_fasttext, X_val_fasttext, y_train_fasttext, y_val_fasttext = fasttext_embedding(train_data, validation_data)
    X_train_word2vec, X_val_word2vec, y_train_word2vec, y_val_word2vec = word2vec_embedding(train_data, validation_data)
    
    # Perform classification using different embeddings
    data_sets = {
        'Count Vector': (X_train_count, X_val_count, y_train_count, y_val_count),
        'TF-IDF': (X_train_tfidf, X_val_tfidf, y_train_tfidf, y_val_tfidf),
        'GloVe': (X_train_glove, X_val_glove, y_train_glove, y_val_glove),
        'FastText': (X_train_fasttext, X_val_fasttext, y_train_fasttext, y_val_fasttext),
        'Word2Vec': (X_train_word2vec, X_val_word2vec, y_train_word2vec, y_val_word2vec)
    }
    
    for data_name, data in data_sets.items():
        X_train, X_val, y_train, y_val = data
        print("\n")
        print(f"Evaluating classifiers on {data_name} embeddings")
        exclude_naive_bayes = data_name in ['GloVe', 'FastText', 'Word2Vec'] 
        # Exclude Naive Bayes for these embeddings due to negative values
        metrics = evaluate_classifiers(X_train, X_val, y_train, y_val, exclude_naive_bayes)
        
        for classifier_name, classifier_metrics in metrics.items():
            print(f"{classifier_name}:")
            for metric_name, metric_value in classifier_metrics.items():
                print(f"  {metric_name}: {metric_value}")


# if __name__ == "__main__":
#     Dataset_path = "/Users/ericwei/Documents/UCL/Postgraduate/ELEC0141 Deep Learning for NLP/Assignment/Assignment_DLNLP/Dataset/"
#     run_evaluation(Dataset_path)
#     run_evaluation(Dataset_path, val = False)

#%%
