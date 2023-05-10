#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 04:56:56 2023

@author: ericwei
"""

import sys
sys.path.append('Data_Preprocessing')
import Data_EDA as de
sys.path.append('Baseline_Model')
import Baseline_Model as bm
sys.path.append('Deep_Learning_Models')
import DL_Model as dl
sys.path.append('Bert_FineTuning')
import Bert_Fine_Tuning as bt
sys.path.append('Hyper-Parameter_Tuning')
import Activation_Tuning as at
import Hyperparameter_Tuning as ht
sys.path.append('Final_Evaluation')
import Final_Evaluation as fe

# Perform Data EDA
Dataset_path = 'Dataset/'
# train_data, val_data, test_data = de.load_dataset(Dataset_path)
# cross_tab = de.perform_eda(train_data, val_data, test_data)

# # Evaluate Machine Learning Algorithms
bm.run_evaluation(Dataset_path)
bm.run_evaluation(Dataset_path, val = False)

# Evaluate Deep Learning Model
all_metrics = dl.run_all_models(Dataset_path)

# Fine Tuning Bert Model
bt.fine_tuning(Dataset_path)

# Activation Tuning
at.evaluate_sigmoid(Dataset_path)
at.evaluate_softmax(Dataset_path)

# Hyper-Parameter Tuning
ht.hyperparameter_tuning(Dataset_path)

# Final Evaluation
fe.Final_Evaluation(Dataset_path, 1e-5, 64)