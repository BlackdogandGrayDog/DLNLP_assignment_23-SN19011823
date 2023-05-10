#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 01:48:16 2023

@author: ericwei
"""
import sys
sys.path.append('../Data_Preprocessing')
import Data_Preprocessing as dp
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU, Dense, SpatialDropout1D, SimpleRNN, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, Add, Input, Layer, Dropout, Flatten
import keras.backend as K
from keras import initializers
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def train_glove_simple_rnn_model(train_data, test_data):
    '''
    This function trains a text classification model using Simple RNN with pre-trained GloVe embeddings.
    It tokenizes and pads the sequences, creates an embedding matrix using GloVe, defines the model architecture, 
    compiles and trains the model, and returns the test accuracy and AUC.
    Additionally, this function plots the ROC curve of the model for the validation data.
    '''

    # Set the maximum sequence length
    max_sequence_length = 200

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data['comment_text'])
    train_sequences = tokenizer.texts_to_sequences(train_data['comment_text'])
    test_sequences = tokenizer.texts_to_sequences(test_data['comment_text'])

    # Pad the sequences
    X_train_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
    X_test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)

    # Create train and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(X_train_padded, train_data['toxicity'], test_size=0.2, random_state=42)

    # Set the embedding dimension
    embedding_dim = 300

    # Load the GloVe embeddings
    embeddings_index = {}
    with open('glove.6B.300d.txt') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray([float(val) for val in values[1:]], dtype='float32')
            embeddings_index[word] = coefs

    # Create the embedding matrix
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Define the model architecture
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
    model.add(SimpleRNN(300, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(1, activation='sigmoid'))
    # Compile and train the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy', AUC(name='auc')])
    model.fit(X_train, y_train, epochs=7, batch_size=32)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test_padded, test_data['toxic'])
    test_accuracy = test_metrics[1]
    test_auc = test_metrics[2]

    # Predict the probabilities for the validation data
    y_pred_prob = model.predict(X_val)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"GloVe Simple RNN (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return test_accuracy, test_auc, val_metrics[1], val_metrics[2]




def train_glove_lstm_model(train_data, test_data):
    '''
    This function trains a text classification model using LSTM with pre-trained GloVe embeddings.
    It takes in the training and test data, tokenizes and pads the sequences, creates an embedding matrix using GloVe,
    defines the model architecture, compiles and trains the model, and then returns the test accuracy and AUC.
    '''
    
    # Set the maximum sequence length
    max_sequence_length = 200
    
    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data['comment_text'])
    train_sequences = tokenizer.texts_to_sequences(train_data['comment_text'])
    test_sequences = tokenizer.texts_to_sequences(test_data['comment_text'])
    
    # Pad the sequences
    X_train_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
    X_test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)
    
    # Create train and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(X_train_padded, train_data['toxicity'], test_size=0.2, random_state=42)
    
    # Set the embedding dimension
    embedding_dim = 300
    
    # Load the GloVe embeddings
    embeddings_index = {}
    with open('glove.6B.300d.txt') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray([float(val) for val in values[1:]], dtype='float32')
            embeddings_index[word] = coefs
    
    # Create the embedding matrix
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
    model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile and train the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy', AUC(name='auc')])
    model.fit(X_train_padded, train_data['toxicity'], epochs=7, batch_size=32)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test_padded, test_data['toxic'])
    test_accuracy = test_metrics[1]
    test_auc = test_metrics[2]

    # Predict the probabilities for the validation data
    y_pred_prob = model.predict(X_val)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"GloVe Simple RNN (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return test_accuracy, test_auc, val_metrics[1], val_metrics[2]





def train_glove_gru_model(train_data, test_data):
    '''
    Introduction: This function trains a text classification model using GRU with pre-trained GloVe embeddings.
    It takes in the training and validation data, tokenizes and pads the sequences, creates an embedding matrix using GloVe,
    defines the model architecture, compiles and trains the model, and then returns the test accuracy.
    '''
 
    # Set the maximum sequence length
    max_sequence_length = 200
    
    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data['comment_text'])
    train_sequences = tokenizer.texts_to_sequences(train_data['comment_text'])
    test_sequences = tokenizer.texts_to_sequences(test_data['comment_text'])
    
    # Pad the sequences
    X_train_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
    X_test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)
    
    # Create train and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(X_train_padded, train_data['toxicity'], test_size=0.2, random_state=42)
    
    # Set the embedding dimension
    embedding_dim = 300
    
    # Load the GloVe embeddings
    embeddings_index = {}
    with open('glove.6B.300d.txt') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray([float(val) for val in values[1:]], dtype='float32')
            embeddings_index[word] = coefs
    
    # Create the embedding matrix
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Define the model architecture
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
    model.add(SpatialDropout1D(0.3))
    model.add(GRU(300))
    model.add(Dense(1, activation='sigmoid'))

    # Compile and train the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy', AUC(name='auc')])
    model.fit(X_train_padded, train_data['toxicity'], epochs=7, batch_size=32)
    val_metrics= model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test_padded, test_data['toxic'])
    test_accuracy = test_metrics[1]
    test_auc = test_metrics[2]

    # Predict the probabilities for the validation data
    y_pred_prob = model.predict(X_val)
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"GloVe GRU (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return test_accuracy, test_auc, val_metrics[1], val_metrics[2]





def train_glove_bidirectional_rnn_model(train_data, test_data):
    '''
    This function trains a text classification model using Bidirectional RNN with pre-trained GloVe embeddings.
    It tokenizes and pads the sequences, creates an embedding matrix using GloVe, defines the model architecture, 
    compiles and trains the model, and returns the test accuracy.
    '''

    # Set the maximum sequence length
    max_sequence_length = 200
    
    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data['comment_text'])
    train_sequences = tokenizer.texts_to_sequences(train_data['comment_text'])
    test_sequences = tokenizer.texts_to_sequences(test_data['comment_text'])
    
    # Pad the sequences
    X_train_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
    X_test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)
    
    # Create train and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(X_train_padded, train_data['toxicity'], test_size=0.2, random_state=42)
    
    # Set the embedding dimension
    embedding_dim = 300
    
    # Load the GloVe embeddings
    embeddings_index = {}
    with open('glove.6B.300d.txt') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray([float(val) for val in values[1:]], dtype='float32')
            embeddings_index[word] = coefs
    
    # Create the embedding matrix
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Define the model architecture
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
    model.add(Bidirectional(SimpleRNN(300, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Dense(1, activation='sigmoid'))

    # Compile and train the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy', AUC(name='auc')])
    model.fit(X_train_padded, train_data['toxicity'], epochs=7, batch_size=32)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test_padded, test_data['toxic'])
    test_accuracy = test_metrics[1]
    test_auc = test_metrics[2]

    # Predict the probabilities for the validation data
    y_pred_prob = model.predict(X_val)
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"GloVe Bidirectional RNN (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return test_accuracy, test_auc, val_metrics[1], val_metrics[2]





def train_bidirectional_gru_lstm_model(train_data, test_data):
    '''
    Introduction: This function trains a text classification model using Bidirectional GRU and LSTM with pre-trained GloVe embeddings.
    It tokenizes and pads the sequences, creates an embedding matrix using GloVe, defines the model architecture, 
    compiles and trains the model, and returns the test accuracy.
    '''

    # Set the maximum sequence length
    max_sequence_length = 200
    
    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data['comment_text'])
    train_sequences = tokenizer.texts_to_sequences(train_data['comment_text'])
    test_sequences = tokenizer.texts_to_sequences(test_data['comment_text'])
    
    # Pad the sequences
    X_train_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
    X_test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)
    
    # Create train and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(X_train_padded, train_data['toxicity'], test_size=0.2, random_state=42)
    
    # Set the embedding dimension
    embedding_dim = 300
    
    # Load the GloVe embeddings
    embeddings_index = {}
    with open('glove.6B.300d.txt') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray([float(val) for val in values[1:]], dtype='float32')
            embeddings_index[word] = coefs
    
    # Create the embedding matrix
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Define the model architecture
    input_layer = Input(shape=(max_sequence_length,))
    x = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False)(input_layer)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x_max = GlobalMaxPooling1D()(x)
    x_avg = GlobalAveragePooling1D()(x)
    x = concatenate([x_max, x_avg])
    
    dense_512 = Dense(512, activation='relu')(x)
    x = Add()([x, dense_512])
    dense_512 = Dense(512, activation='relu')(x)
    x = Add()([x, dense_512])
    output_layer = Dense(1, activation='sigmoid')(x)

    # Compile and train the model
    model = Model(inputs=input_layer, outputs=output_layer)
    # Compile and train the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy', AUC(name='auc')])
    model.fit(X_train_padded, train_data['toxicity'], epochs=7, batch_size=32)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test_padded, test_data['toxic'])
    test_accuracy = test_metrics[1]
    test_auc = test_metrics[2]

    # Predict the probabilities for the validation data
    y_pred_prob = model.predict(X_val)
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"Bidirectional GRU-LSTM (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return test_accuracy, test_auc, val_metrics[1], val_metrics[2]




class AttentionWeightedAverage(Layer):
    """
    The AttentionWeightedAverage layer computes the attention-weighted average
    of the input features using a context vector.

    The layer learns the context vector during training and uses it to compute
    the attention weights for the input features. This helps the model focus on
    the most relevant parts of the input when making predictions.
    """

    def __init__(self, **kwargs):
        self.init = initializers.get('uniform')
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add the context vector as a trainable weight
        self.context_vector = self.add_weight(shape=(input_shape[-1], 1),
                                              name='context_vector',
                                              initializer=self.init,
                                              trainable=True)
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # Compute the attention weights by taking the dot product of the input
        # features with the context vector and applying the softmax function
        attention_weights = K.softmax(K.dot(x, self.context_vector), axis=1)
        
        # Compute the attention-weighted average of the input features
        return K.sum(x * attention_weights, axis=1)

    def compute_output_shape(self, input_shape):
        # The output shape is the same as the input shape, except for the
        # sequence dimension (axis 1), which is reduced to a single element
        return input_shape[0], input_shape[-1]

    def get_config(self):
        # Inherit the base configuration
        base_config = super(AttentionWeightedAverage, self).get_config()
        return dict(list(base_config.items()))

    

    
def train_bidirectional_gru_lstm_attention_model(train_data, test_data):
    '''
    This function trains a text classification model using Bidirectional GRU and LSTM with pre-trained GloVe embeddings and an attention mechanism.
    The attention mechanism helps the model focus on the most important parts of the input sequence, improving its performance.
    The model consists of an embedding layer, a bidirectional GRU layer, a bidirectional LSTM layer, an attention layer, and several dense layers with dropout.
    '''
    # Set the maximum sequence length
    max_sequence_length = 200
    
    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data['comment_text'])
    train_sequences = tokenizer.texts_to_sequences(train_data['comment_text'])
    test_sequences = tokenizer.texts_to_sequences(test_data['comment_text'])
    
    # Pad the sequences
    X_train_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
    X_test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)
    
    # Create train and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(X_train_padded, train_data['toxicity'], test_size=0.2, random_state=42)
    
    # Set the embedding dimension
    embedding_dim = 300
    
    # Load the GloVe embeddings
    embeddings_index = {}
    with open('glove.6B.300d.txt') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray([float(val) for val in values[1:]], dtype='float32')
            embeddings_index[word] = coefs
    
    # Create the embedding matrix
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Define the model architecture
    input_layer = Input(shape=(max_sequence_length,))
    x = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False)(input_layer)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = AttentionWeightedAverage()(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    output_layer = Dense(1, activation='sigmoid')(x)

    # Compile and train the model
    model = Model(inputs=input_layer, outputs=output_layer)
    # Compile and train the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy', AUC(name='auc')])
    model.fit(X_train_padded, train_data['toxicity'], epochs=7, batch_size=32)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test_padded, test_data['toxic'])
    test_accuracy = test_metrics[1]
    test_auc = test_metrics[2]

    # Predict the probabilities for the validation data
    y_pred_prob = model.predict(X_val)
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"GloVe Bidirectional GRU-LSTM with Attention (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return test_accuracy, test_auc, val_metrics[1], val_metrics[2]




def run_all_models(Dataset_path):
    model_functions = [
        train_glove_simple_rnn_model,
        train_glove_lstm_model,
        train_glove_gru_model,
        train_glove_bidirectional_rnn_model,
        train_bidirectional_gru_lstm_model,
        train_bidirectional_gru_lstm_attention_model
    ]

    all_metrics = {}

    for model_func in model_functions:
        train_data, validation_data, test_data = dp.preprocess_datasets(Dataset_path, lang = False)
        test_accuracy, test_auc, val_accuracy, val_auc = model_func(train_data, validation_data)
        all_metrics[model_func.__name__] = {'test_accuracy': test_accuracy, 'test_auc': test_auc, 'val_accuracy': val_accuracy, 'val_auc': val_auc}

    print("Model Name: Test Accuracy | Test AUC | Validation Accuracy | Validation AUC:")
    for model_name, metrics in all_metrics.items():
        print(f"{model_name}: {metrics['test_accuracy']} | {metrics['test_auc']} | {metrics['val_accuracy']} | {metrics['val_auc']} \n")

    return all_metrics
