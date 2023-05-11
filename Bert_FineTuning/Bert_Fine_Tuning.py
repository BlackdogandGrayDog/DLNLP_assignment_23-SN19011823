import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Bidirectional, LSTM, GRU
from tensorflow.keras.metrics import AUC
from transformers import TFBertModel, BertTokenizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc
import sys
sys.path.append('../Data_Preprocessing')
import Data_Preprocessing as dp
sys.path.append('../Deep_Learning_Models')
import DL_Model as dl


def tokenize_data(tokenizer, text_data, max_length):
    '''
    This function tokenizes and pads text data.
    '''
    # Initialize input_ids and attention_masks lists
    input_ids, attention_masks = [], []

    # Iterate over the text data
    for text in text_data:
        # Encode text using the tokenizer
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='tf'
        )

        # Append the input_ids and attention_mask to their respective lists
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Concatenate input_ids and attention_masks tensors
    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)

    return input_ids, attention_masks

def create_tf_dataset(input_ids, attention_masks, labels, batch_size):
    '''
    This function creates a TensorFlow dataset from input_ids, attention_masks, and labels.
    '''
    # Convert input_ids, attention_masks, and labels to tensors
    input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    attention_masks = tf.convert_to_tensor(attention_masks, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    labels = tf.one_hot(labels, depth=2)

    # Create a dataset from the tensors
    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks}, labels))
    dataset = dataset.shuffle(buffer_size=len(input_ids))
    dataset = dataset.batch(batch_size)

    return dataset

def fine_tune_bert_model(model, train_input_ids, train_attention_masks, train_labels, val_input_ids, val_attention_masks, val_labels, batch_size, epochs):
    '''
    This function fine-tunes the BERT model with the given training and validation data.
    '''
    # Compile the model with an optimizer, loss function, and evaluation metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', AUC(name='auc')])

    # Create training and validation datasets
    train_dataset = create_tf_dataset(train_input_ids, train_attention_masks, train_labels, batch_size)
    val_dataset = create_tf_dataset(val_input_ids, val_attention_masks, val_labels, batch_size)

    # Fine-tune the model
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    return model, history

def create_bert_classification_model_basic(model_name, num_labels):
    '''
    This function creates a BERT classification model with one dense layer.
    '''
    # Load the pre-trained BERT model
    base_bert_model = TFBertModel.from_pretrained(model_name)
    base_bert_model.trainable = False
    
    # Define input layers
    input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')
    
    # Pass input through the base BERT model
    bert_output = base_bert_model([input_ids, attention_mask])[0]

    # Add dense layers
    flattened_output = tf.keras.layers.Flatten()(bert_output)
    output = Dense(num_labels, activation='softmax')(flattened_output)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    return model


def create_bert_classification_model_Dense(model_name, num_labels):
    '''
    This function creates a BERT classification model with additional dense layers.
    '''
    # Load the pre-trained BERT model
    base_bert_model = TFBertModel.from_pretrained(model_name)
    base_bert_model.trainable = False

    # Define input layers
    input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')
    
    # Pass input through the base BERT model
    bert_output = base_bert_model([input_ids, attention_mask])[0]
    
    # Add dense layers
    flattened_output = tf.keras.layers.Flatten()(bert_output)
    dense_layer1 = Dense(512, activation='relu')(flattened_output)
    dense_layer2 = Dense(128, activation='relu')(dense_layer1)
    dense_layer3 = Dense(64, activation='relu')(dense_layer2)
    output = Dense(num_labels, activation='softmax')(dense_layer3)
    
    # Create and return the model
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    
    return model


def create_bert_classification_model_LSTM(model_name, num_labels):
    '''
    This function creates a BERT classification model with an added Bidirectional LSTM layer.
    '''
    # Load the pre-trained BERT model
    base_bert_model = TFBertModel.from_pretrained(model_name)
    base_bert_model.trainable = False
    
    # Define input layers
    input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')
    
    # Pass input through the base BERT model
    bert_output = base_bert_model([input_ids, attention_mask])[0]
    
    # Add a Bidirectional LSTM layer
    lstm_output = Bidirectional(LSTM(128, return_sequences=True))(bert_output)
    flattened_output = tf.keras.layers.Flatten()(lstm_output)
    dense_layer1 = Dense(512, activation='relu')(flattened_output)
    dense_layer2 = Dense(128, activation='relu')(dense_layer1)
    dense_layer3 = Dense(64, activation='relu')(dense_layer2)
    output = Dense(num_labels, activation='softmax')(dense_layer3)
    
    # Create and return the model
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    
    return model


def create_bert_classification_model_GRU(model_name, num_labels):
    '''
    This function creates a BERT classification model with added Bidirectional LSTM and GRU layers.
    '''
    # Load the pre-trained BERT model
    base_bert_model = TFBertModel.from_pretrained(model_name)
    base_bert_model.trainable = False
    
    # Define input layers
    input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')
    # Pass input through the base BERT model
    bert_output = base_bert_model([input_ids, attention_mask])[0]
    
    # Add a Bidirectional GRU layer
    gru_output = Bidirectional(GRU(128, return_sequences=True))(bert_output)
    
    # Add a Bidirectional LSTM layer
    lstm_output = Bidirectional(LSTM(128, return_sequences=True))(gru_output)
    
    flattened_output = tf.keras.layers.Flatten()(lstm_output)
    dense_layer1 = Dense(512, activation='relu')(flattened_output)
    dense_layer2 = Dense(128, activation='relu')(dense_layer1)
    dense_layer3 = Dense(64, activation='relu')(dense_layer2)
    output = Dense(num_labels, activation='softmax')(dense_layer3)
    
    # Create and return the model
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    
    return model

def create_bert_classification_model_GRU_with_attention(model_name, num_labels):
    '''
    This function creates a BERT classification model with added Bidirectional LSTM and GRU layers,
    and includes an AttentionWeightedAverage layer.
    '''
    # Load the pre-trained BERT model
    base_bert_model = TFBertModel.from_pretrained(model_name)
    base_bert_model.trainable = False
    
    # Define input layers
    input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')
    # Pass input through the base BERT model
    bert_output = base_bert_model([input_ids, attention_mask])[0]
    
    # Add a Bidirectional GRU layer
    gru_output = Bidirectional(GRU(128, return_sequences=True))(bert_output)
    
    # Add a Bidirectional LSTM layer
    lstm_output = Bidirectional(LSTM(128, return_sequences=True))(gru_output)
    
    # Add an AttentionWeightedAverage layer
    attention_output = dl.AttentionWeightedAverage()(lstm_output)
    
    flattened_output = tf.keras.layers.Flatten()(attention_output)
    dense_layer1 = Dense(512, activation='relu')(flattened_output)
    dense_layer2 = Dense(128, activation='relu')(dense_layer1)
    dense_layer3 = Dense(64, activation='relu')(dense_layer2)
    output = Dense(num_labels, activation='softmax')(dense_layer3)
    
    # Create and return the model
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    
    return model


def plot_confusion_matrix(y_true, y_pred, labels, title):
    '''
    This function plots a confusion matrix given true and predicted labels.
    '''
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(5, 5))
    sns.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt="d", cbar=False)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

def plot_roc_curve(y_true, y_pred_prob, title):
    '''
    This function plots the Receiver Operating Characteristic (ROC) curve given true labels and predicted probabilities.
    '''
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_prob, title):
    '''
    This function plots the Precision-Recall curve given true labels and predicted probabilities.
    '''
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()

def plot_history(history):
    '''
    This function plots training and validation loss, accuracy, and AUC vs. epochs.
    '''
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs = axs.ravel()

    for i, metric in enumerate(["loss", "accuracy"]):
        axs[i].plot(history.history[metric], label="train")
        axs[i].plot(history.history["val_" + metric], label="val")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel(metric)
        axs[i].set_title(f"{metric.capitalize()} vs. Epoch")
        axs[i].legend()

    axs[2].plot(history.history["auc"], label="train")
    axs[2].plot(history.history["val_auc"], label="val")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("AUC")
    axs[2].set_title("AUC vs. Epoch")
    axs[2].legend()
    
    plt.show()


def fine_tuning(Dataset_path):
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

    train_input_ids, train_attention_masks = tokenize_data(tokenizer, train_data['comment_text'].tolist(), max_length)
    val_input_ids, val_attention_masks = tokenize_data(tokenizer, val_data['comment_text'].tolist(), max_length)

    # Set the number of labels, batch size, and number of epochs
    num_labels = 2
    batch_size = 8
    epochs = 10

    # Define the BERT classification models to be fine-tuned
    model_constructors = [
        ('Bert Base Model', create_bert_classification_model_basic),
        ('Bert Dense Model', create_bert_classification_model_Dense),
        ('Bert LSTM Model', create_bert_classification_model_LSTM),
        ('Bert LSTM GRU Model', create_bert_classification_model_GRU),
        ('Bert LSTM GRU Model Attention', create_bert_classification_model_GRU_with_attention)
    ]

    # Fine-tune each model, evaluate its performance on the test dataset, and plot evaluation metrics
    for model_name, model_constructor in model_constructors:
        print(f"\nEvaluating {model_name}\n")
        model = model_constructor(basemodel_name, num_labels)
        fine_tuned_model, history = fine_tune_bert_model(model, train_input_ids, train_attention_masks, train_data['toxic'].tolist(), val_input_ids, val_attention_masks, val_data['toxic'].tolist(), batch_size, epochs)
        plot_history(history)

        # Tokenize and predict for the test dataset
        test_input_ids, test_attention_masks = tokenize_data(tokenizer, test_data['comment_text'].tolist(), max_length)
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
        plot_confusion_matrix(test_true_labels, test_predicted_labels, labels, f'{model_name} Test Confusion Matrix')
        plot_roc_curve(test_true_labels, test_predictions[:, 1], f'{model_name} Test ROC Curve')
        plot_precision_recall_curve(test_true_labels, test_predictions[:, 1], f'{model_name} Test Precision-Recall Curve')