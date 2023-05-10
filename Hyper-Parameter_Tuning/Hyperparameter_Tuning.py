#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 23:55:30 2023

@author: ericwei
"""
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from tensorflow.keras.metrics import AUC
import sys
sys.path.append('../Data_Preprocessing')
import Data_Preprocessing as dp
sys.path.append('../Bert_FineTuning')
import Bert_Fine_Tuning as bt


def fine_tune_bert_model_lr(model, train_input_ids, train_attention_masks, train_labels, val_input_ids, val_attention_masks, val_labels, batch_size, epochs, lr):
    '''
    This function fine-tunes the BERT model with the given training and validation data.
    '''
    # Compile the model with an optimizer, loss function, and evaluation metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', AUC(name='auc')])

    # Create training and validation datasets
    train_dataset = bt.create_tf_dataset(train_input_ids, train_attention_masks, train_labels, batch_size)
    val_dataset = bt.create_tf_dataset(val_input_ids, val_attention_masks, val_labels, batch_size)

    # Fine-tune the model
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    return model, history



def lrbs_tuning(Dataset_path, learning_rate, batch_size):
    """
    Fine-tunes a pre-trained BERT model on a binary classification task to classify text as toxic or non-toxic,
    and evaluates the performance of the fine-tuned model on a test dataset.
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

    # Set the number of labels and number of epochs
    num_labels = 2
    epochs = 10

    # Define the BERT classification model to be fine-tuned
    print("\nEvaluating Bert Dense Model for lr and bs...\n")
    model = bt.create_bert_classification_model_Dense(basemodel_name, num_labels)
    fine_tuned_model, history = fine_tune_bert_model_lr(model, train_input_ids, train_attention_masks, train_data['toxic'].tolist(), val_input_ids, val_attention_masks, val_data['toxic'].tolist(), batch_size, epochs, learning_rate)
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
    bt.plot_confusion_matrix(test_true_labels, test_predicted_labels, labels, 'Bert LSTM GRU Model Attention Test Confusion Matrix')
    bt.plot_roc_curve(test_true_labels, test_predictions[:, 1], 'Bert LSTM GRU Model Attention Test ROC Curve')
    bt.plot_precision_recall_curve(test_true_labels, test_predictions[:, 1], 'Bert LSTM GRU Model Attention Test Precision-Recall Curve')



def hyperparameter_tuning(Dataset_path):
    '''
    This function performs a hyperparameter search over different combinations of learning rates and batch sizes.
    '''
    learning_rates = [1e-3, 1e-4, 1e-5]
    batch_sizes = [16, 32, 64]

    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\nTraining with learning rate: {lr} and batch size: {bs}\n")
            lrbs_tuning(Dataset_path, lr, bs)