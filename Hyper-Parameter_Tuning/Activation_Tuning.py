#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 22:15:48 2023

@author: ericwei
"""
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.metrics import AUC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../Data_Preprocessing')
import Data_Preprocessing as dp
sys.path.append('../Bert_FineTuning')
import Bert_Fine_Tuning as bt


def create_sigmoid_dataset(input_ids, attention_masks, labels, batch_size):
    """
    This function creates a TensorFlow dataset from input_ids, attention_masks, and labels for sigmoid, no one hot encoding.
    """

    # Convert input_ids, attention_masks, and labels to TensorFlow tensors
    input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    attention_masks = tf.convert_to_tensor(attention_masks, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    # Create a TensorFlow dataset from the tensors
    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks}, labels))

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=len(input_ids))

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    return dataset



def fine_tune_bert_model_sigmoid(model, train_input_ids, train_attention_masks, train_labels, val_input_ids, val_attention_masks, val_labels, batch_size, epochs):
    '''
    This function fine-tunes the BERT model with the given training and validation data using signmoid.
    '''
    # Compile the model with an optimizer, loss function, and evaluation metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', AUC(name='auc')])

    # Create training and validation datasets
    train_dataset = create_sigmoid_dataset(train_input_ids, train_attention_masks, train_labels, batch_size)
    val_dataset = create_sigmoid_dataset(val_input_ids, val_attention_masks, val_labels, batch_size)

    # Fine-tune the model
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    return model, history


def create_bert_classification_model_sigmoid(model_name):
    '''
    This function create a bert model using sigmoid activation.
    '''
    # Load the pre-trained BERT model
    base_bert_model = TFBertModel.from_pretrained(model_name)
    base_bert_model.trainable = False
    
    # Define input layers
    input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')
    # Pass input through the base BERT model
    bert_output = base_bert_model([input_ids, attention_mask])[0]
    
    flattened_output = tf.keras.layers.Flatten()(bert_output)
    dense_layer1 = Dense(512, activation='relu')(flattened_output)
    dense_layer2 = Dense(128, activation='relu')(dense_layer1)
    dense_layer3 = Dense(64, activation='relu')(dense_layer2)
    output = Dense(1, activation='sigmoid')(dense_layer3)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    return model


def evaluate_sigmoid(Dataset_path):
    """
    evaluate a pre-trained BERT model on a binary classification task to classify text as toxic or non-toxic,
    and evaluates the performance of the fine-tuned model on a test dataset using sigmoid.
    """
    
    train_data, test_data, _ = dp.preprocess_datasets(Dataset_path, lang=False)

    # Split the training data to training and validation
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42, stratify=train_data['toxicity'])

    text_data = train_data['comment_text'].tolist()
    labels = train_data['toxicity'].tolist()

    max_length = 128
    model_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    input_ids, attention_masks = bt.tokenize_data(tokenizer, text_data, max_length)
    val_input_ids, val_attention_masks = bt.tokenize_data(tokenizer, val_data['comment_text'].tolist(), max_length)

    model = create_bert_classification_model_sigmoid(model_name)

    batch_size = 8
    epochs = 10
    fine_tuned_model, history = fine_tune_bert_model_sigmoid(model, input_ids, attention_masks, labels, val_input_ids, val_attention_masks, val_data['toxicity'].tolist(), batch_size, epochs)

    # Plot training history
    bt.plot_history(history)

    # Tokenize and predict for the test dataset
    test_input_ids, test_attention_masks = bt.tokenize_data(tokenizer, test_data['comment_text'].tolist(), max_length)
    test_predictions = fine_tuned_model.predict([test_input_ids, test_attention_masks], batch_size=batch_size)

    # Evaluate the model
    test_true_labels = test_data['toxic'].tolist()
    test_true_labels_tensor = tf.convert_to_tensor(test_true_labels, dtype=tf.float32)
    test_loss, test_accuracy, test_auc = fine_tuned_model.evaluate([test_input_ids, test_attention_masks], test_true_labels_tensor, batch_size=batch_size)

    # Calculate additional metrics
    test_predicted_labels = (test_predictions > 0.5).astype(int).flatten()
    test_precision = precision_score(test_true_labels, test_predicted_labels)
    test_recall = recall_score(test_true_labels, test_predicted_labels)
    test_f1 = f1_score(test_true_labels, test_predicted_labels)
    test_roc_auc = roc_auc_score(test_true_labels, test_predictions)

    # Plot confusion matrix, ROC curve, and PR curve
    bt.plot_confusion_matrix(test_true_labels, test_predicted_labels, [0, 1], 'Test Confusion Matrix')
    bt.plot_roc_curve(test_true_labels, test_predictions, 'Test ROC Curve')
    bt.plot_precision_recall_curve(test_true_labels, test_predictions, 'Test PR Curve')

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    print("Test AUC:", test_auc)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test F1 Score:", test_f1)
    print("Test ROC AUC Score:", test_roc_auc)




def evaluate_softmax(Dataset_path):
    """
    evaluate a pre-trained BERT model on a binary classification task to classify text as toxic or non-toxic,
    and evaluates the performance of the fine-tuned model on a test dataset using softmax.
    """
    
    # Preprocess the dataset
    train_data, test_data, _ = dp.preprocess_datasets(Dataset_path, lang=False)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Tokenize the data using BERT tokenizer
    max_length = 128
    basemodel_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(basemodel_name)

    train_input_ids, train_attention_masks = bt.tokenize_data(tokenizer, train_data['comment_text'].tolist(), max_length)
    val_input_ids, val_attention_masks = bt.tokenize_data(tokenizer, val_data['comment_text'].tolist(), max_length)

    # Set the number of labels, batch size, and number of epochs
    num_labels = 2
    batch_size = 32
    epochs = 10

    # Define the BERT classification models to be fine-tuned
    model_constructors = [
        ('Softmax', bt.create_bert_classification_model_Dense)
    ]

    # Fine-tune each model, evaluate its performance on the test dataset, and plot evaluation metrics
    for model_name, model_constructor in model_constructors:
        print(f"\nEvaluating {model_name}\n")
        model = model_constructor(basemodel_name, num_labels)
        fine_tuned_model, history = bt.fine_tune_bert_model(model, train_input_ids, train_attention_masks, train_data['toxic'].tolist(), val_input_ids, val_attention_masks, val_data['toxic'].tolist(), batch_size, epochs)
        bt.plot_history(history)

        # Tokenize and predict for the test dataset
        test_input_ids, test_attention_masks = bt.tokenize_data(tokenizer, test_data['comment_text'].tolist(), max_length)
        test_logits = fine_tuned_model.predict([test_input_ids, test_attention_masks], batch_size=batch_size)
        test_predictions = tf.nn.softmax(test_logits, axis=-1).numpy()

        # Evaluate the model
        test_true_labels = test_data['toxic'].tolist()
        test_true_labels = tf.convert_to_tensor(test_true_labels, dtype=tf.int32)
        test_true_labels_one_hot = tf.one_hot(test_true_labels, depth=2)  # Convert labels to one-hot encoded vectors
        test_loss, test_accuracy, test_auc = fine_tuned_model.evaluate([test_input_ids, test_attention_masks], test_true_labels_one_hot, batch_size=batch_size)

        # Set the threshold and calculate predicted labels
        threshold = 0.5
        test_predicted_labels = (test_predictions[:, 1] > threshold).astype(int)

        # Calculate evaluation metrics
        test_precision = precision_score(test_true_labels, test_predicted_labels)
        test_recall = recall_score(test_true_labels, test_predicted_labels)
        test_f1 = f1_score(test_true_labels, test_predicted_labels)
        test_roc_auc = roc_auc_score(test_true_labels, test_predictions[:, 1])

        # Print results
        print("Test Loss:", test_loss)
        print("Test AUC:", test_auc)
        print("Test Accuracy:", test_accuracy)
        print("Test Precision:", test_precision)
        print("Test Recall:", test_recall)
        print("Test F1 Score:", test_f1)
        print("Test ROC AUC Score:", test_roc_auc)
        
        # Plot confusion matrix, ROC curve, and precision-recall curve
        labels = [0, 1]
        bt.plot_confusion_matrix(test_true_labels, test_predicted_labels, labels, f'{model_name} Test Confusion Matrix')
        bt.plot_roc_curve(test_true_labels, test_predictions[:, 1], f'{model_name} Test ROC Curve')
        bt.plot_precision_recall_curve(test_true_labels, test_predictions[:, 1], f'{model_name} Test Precision-Recall Curve')