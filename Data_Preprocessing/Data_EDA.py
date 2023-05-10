#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 22:13:29 2023

@author: ericwei
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from langdetect import detect
from collections import Counter
from pycountry import languages
from tqdm import tqdm
from better_profanity import profanity
import re
#%%
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'


def language_full_name(alpha2_code):
    try:
        return languages.get(alpha_2=alpha2_code).name
    except:
        return 'unknown'


def load_dataset(Dataset_path):
    '''
    This function takes a dataset path as input and returns train, validation, and test dataframes.
    '''
    test_path = Dataset_path + "test.csv"
    validation_path = Dataset_path + "validation.csv"
    train_path = Dataset_path + "jigsaw-toxic-comment-train.csv"

    test_data = pd.read_csv(test_path)
    val_data = pd.read_csv(validation_path)
    train_data = pd.read_csv(train_path)

    # Detect language for train and validation datasets
    tqdm.pandas()
    train_data['lang'] = train_data['comment_text'].progress_apply(detect_language)
    train_data['lang_full'] = train_data['lang'].progress_apply(language_full_name)

    # Add 'English' column to train_data
    train_data['English'] = train_data['lang'].progress_apply(lambda x: 1 if x == 'en' else 0)

    return train_data, val_data, test_data


def plot_target_distribution(df, title):
    '''
    This function takes a dataframe, a title string as input, and plots two histograms of the binary toxicity score distribution,
    one for "Toxic" scores (above 0.5) and one for "Non-Toxic" scores (below or equal to 0.5).
    '''
    plt.figure(figsize=(8, 6))

    target_column = 'toxic'
        
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



def plot_target_distribution_by_language(df, title):
    unique_languages = df['lang'].unique()
    n_languages = len(unique_languages)
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'orange', 'gray', 'brown']

    plt.figure(figsize=(10, 6))

    target_column = 'toxic'
    width = 0.8 / n_languages

    toxic_counts = []
    non_toxic_counts = []

    for idx, lang in enumerate(unique_languages):
        lang_df = df[df['lang'] == lang]

        # separate the "Toxic" and "Non-Toxic" scores into two lists
        toxic_scores = lang_df[lang_df[target_column] > 0.5][target_column]
        non_toxic_scores = lang_df[lang_df[target_column] <= 0.5][target_column]

        toxic_counts.append(len(toxic_scores))
        non_toxic_counts.append(len(non_toxic_scores))

        # plot the "Toxic" histogram with different colors
        plt.bar(1 + idx * width, len(toxic_scores), width, color=colors[idx % len(colors)], label=f'Toxic - {lang}')

        # plot the "Non-Toxic" histogram with different colors
        plt.bar(idx * width, len(non_toxic_scores), width, color=colors[idx % len(colors)], alpha=0.6, label=f'Non-Toxic - {lang}')

    plt.ylabel('Frequency', fontweight='bold')
    plt.xticks([0.2, 1.2], ['Non-Toxic', 'Toxic'], fontweight='bold')
    plt.legend(loc='upper right', title='Languages', ncol=n_languages, fontsize='small', title_fontsize='medium')
    plt.title(title, fontweight='bold')
    plt.show()




def plot_toxic_classification_distribution(df, title):
    """
    This function takes a dataframe and a title string as input,
    and plots a bar chart of the frequency of different toxic classifications in the training dataset.
    """
    toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    counts = df[toxic_columns].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=counts.index, y=counts.values)
    plt.xlabel('Toxic Classifications', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.show()


def plot_english_distribution(df, title, pie_chart=False):
    '''
    This function takes a dataframe, a title string, and a boolean flag as input.
    If the pie_chart flag is True, it plots a pie chart of the English and Non-English distribution.
    Otherwise, it plots a bar chart of the English and Non-English distribution.
    '''
    english_counts = df['English'].value_counts()

    if pie_chart:
        labels = ['English', 'Non-English']
        sizes = english_counts.values
        plt.figure(figsize=(10, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(title, fontweight='bold')
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='English')
        plt.xlabel('Language', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.title(title, fontweight='bold')
        plt.xticks([0, 1], ['Non-English', 'English'], fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.show()


def plot_language_distribution(df, title, pie_chart=False):
    '''
    This function takes a dataframe, a title string, and a boolean flag as input.
    If the pie_chart flag is True, it plots a pie chart of the language distribution.
    Otherwise, it plots a bar chart of the language distribution.
    '''
    df['lang_full'] = df['lang'].apply(language_full_name)
    language_counts = Counter(df['lang_full'])

    if pie_chart:
        labels, sizes = zip(*language_counts.items())
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(title, fontweight='bold')
        plt.show()
    else:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='lang_full')
        plt.xlabel('Language', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.title(title, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.show()


def plot_text_length_distribution(df, title, column_name='comment_text'):
    '''
    This function takes a dataframe, a title string, and a column name as input,
    and plots a histogram of the text length distribution.
    '''
    df['text_length'] = df[column_name].apply(lambda x: len(x.split()))
    plt.figure(figsize=(8, 6))
    sns.histplot(df['text_length'], kde=False, bins=50)
    plt.xlabel('Text Length', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.show()


def generate_word_cloud(df, title='', is_toxic=True, column_name='comment_text'):
    '''
    This function takes a dataframe, a title string, a boolean flag for whether to generate a word cloud for toxic comments,
    a boolean flag for whether the dataframe is train_data, and a column name as input.
    It generates a word cloud based on the comments in the dataframe.
    '''
    target_column = 'toxic'

    if is_toxic:
        text = " ".join(comment for comment in df[df[target_column] > 0.5][column_name])
    else:
        text = " ".join(comment for comment in df[df[target_column] <= 0.5][column_name])

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=300, min_font_size=10).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontweight='bold')
    plt.show()
    
    
    
def plot_toxic_classification_vs_text_length(df):
    """
    This function takes a dataframe as input and creates two subplots:
    1. Histograms of frequency of each toxic classification versus text length.
    2. Correlation or trend of toxicity in the training dataset versus text length.
    """
    toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['text_length'] = df['comment_text'].apply(lambda x: len(x.split()))
    
    # Subplot 1: Histograms of frequency of each toxic classification versus text length
    plt.figure(figsize=(12, 8))
    colors = ['b', 'r', 'g', 'y', 'm', 'c']
    
    for i, column in enumerate(toxic_columns):
        sns.histplot(data=df[df[column] == 1], x='text_length', bins=50, color=colors[i], kde=True, alpha=0.5, label=column.capitalize())
    
    plt.title('Frequency of Each Toxic Classification vs Text Length', fontweight='bold')
    plt.xlabel('Text Length', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.legend()
    plt.show()
    
    # Subplot 2: Correlation or trend of toxicity in the training dataset versus text length
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x='text_length', y='toxic', scatter_kws={'alpha': 0.3})
    plt.title('Correlation of Toxicity vs Text Length', fontweight='bold')
    plt.xlabel('Text Length', fontweight='bold')
    plt.ylabel('Toxicity', fontweight='bold')
    plt.show()


def create_cross_tabulation(df):
    '''
    This function takes a dataframe as input and creates a single cross-tabulation table between 'toxic' and each of the other classifications.
    '''
    classification_columns = df.columns[2:-5]

    # Convert classification_columns to numeric data type
    df[classification_columns] = df[classification_columns].apply(pd.to_numeric, errors='coerce')

    # Initialize an empty DataFrame to store the results
    cross_tab = pd.DataFrame()

    # Create binary versions of 'toxic' and other classification columns
    df['toxic_bin'] = (df['toxic'] > 0.5).astype(int)

    for col in classification_columns:
        df[col+'_bin'] = (df[col] > 0.5).astype(int)
        temp_cross_tab = pd.crosstab(df['toxic_bin'], df[col+'_bin'])
        cross_tab[f'{col}_0'] = temp_cross_tab.iloc[:, 0]
        
        if temp_cross_tab.shape[1] > 1:
            cross_tab[f'{col}_1'] = temp_cross_tab.iloc[:, 1]
        else:
            cross_tab[f'{col}_1'] = 0

    cross_tab.index.name = 'Toxic'
    return cross_tab


def plot_text_length_histogram(df, dataset_title):
    '''
    This function takes a dataframe and dataset_title as input and creates a histogram plot showing the distribution of text length for toxic and non-toxic comments.
    '''
    # Convert the 'toxic' column to binary values
    df['toxic_bin'] = (df['toxic'] > 0.5).astype(int)

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
    
    
def plot_toxic_classification_frequency_histogram(df, title):
    """
    This function takes a dataframe as input and creates a histogram showing the frequency of toxic classifications.
    """
    classification_columns = df.columns[2:-5]
    counts = df[classification_columns].apply(lambda x: x > 0.5).sum()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts.index, y=counts.values)
    plt.xlabel('Toxic Classifications')
    plt.ylabel('Frequency')
    plt.title(title, fontweight='bold')
    plt.show()


def plot_multiple_labels_histogram(df, title):
    """
    This function takes a dataframe as input and creates a histogram showing the number of multiple labels in a comment.
    """
    classification_columns = df.columns[2:-5]
    multiple_labels_count = df[classification_columns].apply(lambda x: x > 0.5).sum(axis=1).value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=multiple_labels_count.index, y=multiple_labels_count.values)
    plt.xlabel('Number of Labels')
    plt.ylabel('Frequency')
    plt.title(title, fontweight='bold')
    plt.xticks(multiple_labels_count.index, ['None'] + [f'{i} Label(s)' for i in multiple_labels_count.index[1:]])
    plt.show()
    


def plot_unique_words_percentage(df, title, column_name='comment_text'):
    """
    This function takes a dataframe as input and creates two line plots showing the percentage of unique words in the comments
    for toxic and non-toxic comments.
    """
    toxic_df = df[df['toxic'] > 0.5]
    nontoxic_df = df[df['toxic'] <= 0.5]

    unique_words_ratio_toxic = toxic_df[column_name].apply(lambda x: len(set(x.split())) / len(x.split()) * 100)
    unique_words_ratio_nontoxic = nontoxic_df[column_name].apply(lambda x: len(set(x.split())) / len(x.split()) * 100)

    plt.figure(figsize=(10, 5))

    # Non-toxic comments line
    nontoxic_counts, nontoxic_bins, _ = plt.hist(unique_words_ratio_nontoxic, bins=40, alpha=0.5, color='green', label='Non-toxic', density=True)
    nontoxic_x = (nontoxic_bins[:-1] + nontoxic_bins[1:]) / 2
    plt.plot(nontoxic_x, nontoxic_counts * 100, color='green')
    plt.fill_between(nontoxic_x, nontoxic_counts * 100, alpha=0.2, color='green')

    # Toxic comments line
    toxic_counts, toxic_bins, _ = plt.hist(unique_words_ratio_toxic, bins=40, alpha=0.3, color='red', label='Toxic', density=True)
    toxic_x = (toxic_bins[:-1] + toxic_bins[1:]) / 2
    plt.plot(toxic_x, toxic_counts * 100, color='red')
    plt.fill_between(toxic_x, toxic_counts * 100, alpha=0.1, color='red')

    plt.xlabel('Percentage of Unique Words')
    plt.ylabel('Percentage of Comments')
    plt.title(title, fontweight='bold')
    plt.legend()
    plt.show()
    
    
def plot_offensive_words_histogram(dataset, column_name):
    """
    This function takes a dataset and a column_name as input, and plots a histogram of the top 20 offensive words
    and their frequencies in the specified column of the dataset.
    """
    
    offensive_words = []
    
    # Load the profanity words
    profanity.load_censor_words()
    
    # Iterate through each comment in the dataset
    for comment in tqdm(dataset[column_name]):
        words = re.findall(r'\w+', comment)
        for word in words:
            if profanity.contains_profanity(word):
                offensive_words.append(word.lower())
    
    # Calculate the frequency of each offensive word
    word_freq = pd.Series(offensive_words).value_counts()
    
    # Plot the histogram
    plt.figure(figsize=(12, 6))
    plt.bar(word_freq.index[:20], word_freq.values[:20])
    plt.xlabel('Offensive Words', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title('Histogram of Offensive Words and Their Frequencies', fontweight='bold')
    plt.xticks(rotation=45)
    plt.show()

    


def perform_eda(train_data, val_data, test_data):
    '''
    This function takes train, validation, and test data
    Perform EDA on train, validation, and test datasets
    '''

    # Plot target distribution for train and validation datasets
    plot_target_distribution(train_data, 'Distribution of Binary Toxicity Scores in Train Data')
    plot_target_distribution(val_data, 'Distribution of Binary Toxicity Scores in Validation Data')
    plot_target_distribution_by_language(val_data, 'Validation Data Toxicity Distribution by Language')
    
    # Plot text length histograms for train and validation datasets
    plot_text_length_histogram(train_data, 'Train Data')
    plot_text_length_histogram(val_data, 'Validation Data')

    # Plot toxic classification vs text length for the training dataset
    plot_toxic_classification_vs_text_length(train_data)
    
    # Plot offensive words training dataset
    plot_offensive_words_histogram(train_data, 'comment_text')
    
    # Loop through the datasets (train, validation, test) and generate plots for each
    for idx, (data, label) in enumerate(zip([train_data, val_data, test_data], ['Train', 'Validation', 'Test'])):
        column_name = 'comment_text' if idx != 2 else 'content'

        # Plot text length distribution
        plot_text_length_distribution(data, f'Distribution of Text Lengths in {label} Data', column_name)

        # Plot language distribution (both bar chart and pie chart)
        plot_language_distribution(data, f'Distribution of Languages in {label} Data')
        plot_language_distribution(data, f'Distribution of Languages in {label} Data (Pie Chart)', pie_chart=True)

        # Generate word clouds for toxic and non-toxic comments (skip for test data)
        if label != 'Test':
            generate_word_cloud(data, title=f'Word Cloud for Toxic Comments in {label} Data', is_toxic=True, column_name=column_name)
            generate_word_cloud(data, title=f'Word Cloud for Non-Toxic Comments in {label} Data', is_toxic=False, column_name=column_name)

    # Plot frequency of different toxic classifications in the training dataset
    plot_toxic_classification_distribution(train_data, 'Frequency of Different Toxic Classifications in Train Data')

    # Plot English and Non-English distribution in the training dataset (both bar chart and pie chart)
    plot_english_distribution(train_data, 'Distribution of English and Non-English Texts in Train Data')
    plot_english_distribution(train_data, 'Distribution of English and Non-English Texts in Train Data (Pie Chart)', pie_chart=True)
    
    plot_toxic_classification_frequency_histogram(train_data, 'Frequency of Toxic Classifications in Train Data')
    plot_multiple_labels_histogram(train_data, 'Number of Multiple Labels in Train Data')
    plot_unique_words_percentage(train_data, 'Percentage of Unique Words in Train Data')

    # Create a cross-tabulation table for toxic and other classifications in the training dataset
    cross_tab = create_cross_tabulation(train_data)
    print(cross_tab)
    
    return cross_tab


#%%
# if __name__ == "__main__":
#     train_data, val_data, test_data = load_dataset(Dataset_path)
#     cross_tab = perform_eda(train_data, val_data, test_data)