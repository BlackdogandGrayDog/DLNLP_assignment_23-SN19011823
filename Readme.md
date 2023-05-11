# Multilingual Toxic Comments Detection using Advanced NLP Techniques

This repository is dedicated to the task of multilingual toxic comment detection, as a part of the [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/overview) competition on Kaggle. We aim to develop a comprehensive and robust solution that can effectively identify and classify toxic comments across multiple languages.

## Introduction

Toxic comment detection is a critical task in ensuring a safe and constructive online environment. In this project, we propose a complete pipeline for multilingual toxic comment detection leveraging advanced natural language processing (NLP) techniques and analyse various techniques. Our solution includes data preprocessing, exploratory data analysis, baseline model creation, deep learning models, fine-tuning using BERT, hyperparameter tuning, and final evaluation.

## Contents

This repository contains various modules that cover different aspects of our multilingual toxic comment detection solution:

1. **Data_Preprocessing**: This module provides functions for loading and preprocessing the dataset. It includes techniques for handling multilingual data, text cleaning, tokenization, and data partitioning.

2. **Data_EDA**: This script conducts exploratory data analysis on the preprocessed data. It provides insights into the dataset's characteristics and the distribution of toxic and non-toxic comments across different languages.

3. **Baseline_Model**: This module constructs and evaluates several baseline models for the task of toxic comment classification. It helps establish a performance benchmark for subsequent advanced models.

4. **DL_Model**: This script builds and trains deep learning models for the task, including CNN, RNN, and LSTM models. It also includes functions for model evaluation and performance visualization.

5. **Bert_Fine_Tuning**: This module is dedicated to the fine-tuning of the BERT model for our specific task. It includes functions for model construction, training, and evaluation.

6. **Hyper-Parameter_Tuning**: This script provides methods for tuning the hyperparameters of the models, using techniques such as grid search and random search.

7. **Final_Evaluation**: This module evaluates the final models' performance and compares them to the baseline models. It includes functions for calculating various performance metrics and visualizing the results.

8. **main**: The main script orchestrates the entire process, from data preprocessing to model training, evaluation, and comparison.

## Dependencies

The project is built using Python 3.9 and requires the following packages:

  - python=3.9
  - pip
  - pip:
    - numpy
    - scipy
    - pandas
    - scikit-learn
    - matplotlib
    - seaborn
    - wordcloud
    - langdetect
    - datetime
    - tensorflow
    - keras
    - tqdm
    - nltk
    - better_profanity
    - emoji
    - gingerit
    - pycountry
    - spyder-kernels
    - keras-applications
    - keras-preprocessing
    - transformers

## Dataset

The dataset used in this project is provided by the competition. You can access it [here](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/data).

## How to Run the Code

Before running the code, make sure to download the required GloVe embeddings file, `glove.6B.300d.txt`, from this link: [https://nlp.stanford.edu/data/glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip). Due to size limitations, the file has not been uploaded to the repository. Place the downloaded `glove.6B.300d.txt` file in the same directory as the `main.py` file. If you plan to run separate files, copy the `glove.6B.300d.txt` file into the `Baseline_Model` and `Deep_Learning_Models` folders.

Also, please ensure downloading all datasets and placed in to Dataset Folder.

### Directory Structure

The repository is organized as follows:

```
|-- main.py
|-- glove.6B.300d.txt (download separately)
|-- Data_Preprocessing
| |-- ...
|-- Baseline_Model
| |-- ...
| |-- glove.6B.300d.txt (download separately)
|-- DL_Model
| |-- ...
|-- Bert_Fine_Tuning
| |-- ...
|-- Hyper-Parameter_Tuning
| |-- ...
|-- Final_Evaluation
| |-- ...
|-- Deep_Learning_Models
| |-- ...
| |-- glove.6B.300d.txt (download separately)
```
