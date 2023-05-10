#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on SAT MAY 1 12:12:27 2023

@author: ericwei
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect
import re
from tqdm import tqdm
import emoji
from gingerit.gingerit import GingerIt
import nltk
from nltk.corpus import stopwords

#%%
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'


def load_dataset(Dataset_path, lang):
    '''
    This function takes a dataset path as input and returns train, validation, and test dataframes.
    '''
    test_path = Dataset_path + "test.csv"
    validation_path = Dataset_path + "validation.csv"
    train_path = Dataset_path + "jigsaw-toxic-comment-train.csv"

    test_data = pd.read_csv(test_path)
    val_data = pd.read_csv(validation_path)
    train_data = pd.read_csv(train_path)
    
    train_data = train_data[0:50000]
    
    if lang:
        # Detect language for train and validation datasets
        tqdm.pandas()
        train_data['lang'] = train_data['comment_text'].progress_apply(detect_language)

    # Add 'toxicity' column to train_data
    toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_data['toxicity'] = train_data[toxic_columns].apply(lambda x: 1 if any(x > 0.5) else 0, axis=1)

    return train_data, val_data, test_data


def plot_target_distribution(df, title):
    '''
    This function takes a dataframe, a title string as input, and plots two histograms of the binary toxicity score distribution,
    one for "Toxic" scores (above 0.5) and one for "Non-Toxic" scores (below or equal to 0.5).
    '''
    plt.figure(figsize=(8, 6))

    target_column = 'toxicity'
        
    # separate the "Toxic" and "Non-Toxic" scores into two lists
    toxic_scores = df[df[target_column] > 0.5][target_column]
    non_toxic_scores = df[df[target_column] <= 0.5][target_column]
    
    # plot the "Toxic" histogram with an orange color
    plt.hist(toxic_scores, bins=2, color='orange', alpha=0.8, edgecolor='black', linewidth=1.2)
    plt.ylabel('Frequency', fontweight='bold')
    
    # plot the "Non-Toxic" histogram with a blue color
    plt.hist(non_toxic_scores, bins=2, color='blue', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # set x-axis label position to the center of the bars
    plt.xticks([0.25, 1.25], ['Non-Toxic', 'Toxic'], fontweight='bold')
    plt.xlim([-0.1, 1.6])
    
    plt.title(title, fontweight='bold')
    plt.show()
    
    
def plot_text_length_histogram(df, dataset_title):
    '''
    This function takes a dataframe and dataset_title as input and creates a histogram plot showing the distribution of text length for toxic and non-toxic comments.
    '''
    # Convert the 'toxic' column to binary values
    df['toxic_bin'] = (df['toxicity'] > 0.5).astype(int)

    # Calculate the length of the text for each comment
    df['text_length'] = df['comment_text'].apply(len)

    # Separate the toxic and non-toxic comments
    toxic_comments = df[df['toxic_bin'] == 1]
    non_toxic_comments = df[df['toxic_bin'] == 0]

    # Create the histogram plot
    plt.figure(figsize=(10, 6))
    sns.histplot(non_toxic_comments['text_length'], color='blue', label='Non-Toxic', bins=50, kde=False)
    sns.histplot(toxic_comments['text_length'], color='red', label='Toxic', bins=50, kde=False)
    plt.legend()
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.title(f'Text Length Distribution of Toxic and Non-Toxic Comments ({dataset_title})', fontweight='bold')
    plt.show()
    


def clean_text(text):
    """
    This function takes a text string as input and performs the following cleaning steps:
    1. Removes HTML tags.
    2. Removes URLs.
    3. Removes IP addresses.
    4. Removes numbers and digits.
    5. Removes all punctuation except ".", "!", "?".
    6. Inserts spaces between groups of lowercase and uppercase characters.
    7. Removes extra spaces.
    8. Removes stopwords.
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    
    # Remove IP addresses
    text = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', ' ', text)
    
    # Remove numbers and digits
    text = re.sub(r'\d+', ' ', text)

    # Remove all punctuation except ".", "!", "?"
    text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
    
    # Insert spaces between groups of lowercase and uppercase characters
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    # Download the stopwords from nltk
    nltk.download('stopwords', quiet=True)
    # Create a set of English stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    text = ' '.join(filtered_words)

    return text



def replace_obscene_words(text):
    """
    This function takes a text string as input, and replaces censored characters in offensive words
    with the original characters.
    """

    # Define a dictionary of censored words and their corresponding uncensored versions
    obscene_words = {
        r"\bf[*]+ck\b": "fuck",
        r"\bf[*]+king\b": "fucking",
        r"\bf[*]+ked\b": "fucked",
        r"\bf[*]+kng\b": "fucking",
        r"\bb[*]+tch\b": "bitch",
        r"\bb[*]+ch\b": "bitch",
        r"\bbi[*]+h\b": "bitch",
        r"\bp[*]+ssy\b": "pussy",
        r"\bp[*]+sy\b": "pussy",
        r"\bp[*]+s[*]+y\b": "pussy",
        r"\bsh[*]+t\b": "shit",
        r"\bs[*]+t\b": "shit",
        r"\bs[*]+[*]+t\b": "shit",
        r"\ba[*]+hole\b": "asshole",
        r"\bdi[*]+k\b": "dick",
        r"\bd[*]+k\b": "dick",
        r"\bd[*]+[*]+k\b": "dick",
        r"\bmor[*]+n\b": "moron",
        r"\bm[*]+n\b": "moron",
        r"\bm[*]+[*]+n\b": "moron",
        r"\bfagg[*]+t\b": "faggot",
        r"\bf[*]+t\b": "faggot",
        r"\bf[*]+[*]+t\b": "faggot",
        r"\bfu[*]+k\b": "fuck",
        r"\bf[*]+k\b": "fuck",
        r"\bf[*]+king\b": "fucking",
    }

    # Replace censored characters in the offensive words using regular expressions
    for censored_word, uncensored_word in obscene_words.items():
        text = re.sub(censored_word, uncensored_word, text, flags=re.IGNORECASE)

    return text



def replace_emojis(text):
    """
    This function takes a text string as input and replaces emojis and emoticons with corresponding words.
    """
    # Replace emoticons with corresponding words
    emoji_dict = {
        ":-(": "sad",
        ":(": "sad",
        ":'(": "crying",
        ":-)": "happy",
        ":)": "happy",
        ":]": "happy",
        ":-D": "laughing",
        ":D": "laughing",
        ";-)": "winking",
        ";)": "winking",
        ":-P": "sticking_tongue_out",
        ":P": "sticking_tongue_out",
        ":-O": "surprised",
        ":O": "surprised",
        ":-|": "neutral",
        ":|": "neutral",
        ":-/": "confused",
        ":/": "confused",
        ":-*": "kissing",
        ":*": "kissing",
        ":-X": "kissing",
        ":X": "kissing",
        ":-\\": "skeptical",
        ":\\": "skeptical",
        ":-(": "sad",
        ">:(": "angry",
        ":-)": "happy",
        ">:)": "evil",
        ":-3": "smirking",
        ":3": "smirking",
    }

    
    for emoji_str, word in emoji_dict.items():
        text = text.replace(emoji_str, word)
    
    # Replace emojis with corresponding words
    text = emoji.demojize(text)
    text = re.sub(r':([a-z_]+):', r'\1', text) # Remove colons

    return text


def correct_grammatical_errors(text):
    """
    This function takes a text string as input and corrects grammatical errors using the GingerIt library.
    """
    parser = GingerIt()
    corrected_text = parser.parse(text)['result']
    return corrected_text



def replace_text(text):
    """
    This function takes a text string as input and performs the following replacements:
    1. Replaces censored characters in offensive words with the original characters.
    2. Replaces sentence end marks with special tokens.
    3. Replaces emojis and emoticons with corresponding words.
    4. Corrects grammatical errors.
    5. Converts all words into lower case.
    """
    
    # Replace obscene words
    text = replace_obscene_words(text)
    
    # Replace emojis and emoticons
    text = replace_emojis(text)
    
    # Correct grammatical errors
    # text = correct_grammatical_errors(text)

    # Replace sentence end marks with special tokens
    text = re.sub(r'!', ' exclmrk ', text)
    text = re.sub(r'\?', ' qstmrk ', text)
    text = re.sub(r'\.', ' eosmkr ', text)
    
    # Convert to lower case
    text = text.lower()

    return text


def preprocess_datasets(dataset_path, lang):
    """
    Function to load the datasets, clean and replace the text in the training data.
    """

    # Load the datasets
    train_data, val_data, test_data = load_dataset(dataset_path, lang)

    # Clean and replace the text in the training data
    tqdm.pandas()
    train_data['comment_text'] = train_data['comment_text'].progress_apply(clean_text)
    train_data['comment_text'] = train_data['comment_text'].progress_apply(replace_text)
    val_data['comment_text'] = val_data['comment_text'].progress_apply(clean_text)
    val_data['comment_text'] = val_data['comment_text'].progress_apply(replace_text)
    
    # Create test dataset manually
    test_data = val_data[:3000]

    # Return the preprocessed datasets
    return train_data, val_data, test_data


#%%
# if __name__ == "__main__":
#     Dataset_path = "/Users/ericwei/Documents/UCL/Postgraduate/ELEC0141 Deep Learning for NLP/Assignment/Assignment_DLNLP/Dataset/"
#     train_data, val_data, test_data = preprocess_datasets(Dataset_path)
    
#     # example_index = 997
#     # original_comment = train_data.loc[example_index, 'comment_text']
#     # print("Original comment:\n", original_comment)
    
#     # cleaned_comment = clean_text(original_comment)
#     # print("\nCleaned comment:\n", cleaned_comment)
    
#     # replaced_comment = replace_text(cleaned_comment)
#     # print("\nReplaced comment:\n", replaced_comment)