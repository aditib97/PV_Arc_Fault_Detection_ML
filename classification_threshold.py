# -*- coding: utf-8 -*-
"""
Classification Threshold Tuning

@author: Aditi Bongale
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import ExtraTreesClassifier

#====================================================
# Function to calculate metrics
#====================================================
def calculate_metrics(y_true, y_prob, thresholds):
    accuracies, fnrs, fprs = [], [], []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        fnr = fn / (fn + tp)  # False Negative Rate
        fpr = fp / (fp + tn)  # False Positive Rate
        
        accuracies.append(accuracy)
        fnrs.append(fnr)
        fprs.append(fpr)
    
    return accuracies, fnrs, fprs

#====================================================
# Read in data
#====================================================
# Paths within directory
combined_dir = 'data/03_CombinedProcessedData_csv'
combined_csv_name = 'Combined_I1_4K.csv'

# Read the CSV file into a DataFrame
input_data = pd.read_csv(os.path.join(combined_dir, combined_csv_name))
print('Tabular data read')

#====================================================
# Separating 3 inputs 
#====================================================
df = input_data.drop(['Target', 'Group', 'Median_i_dc', 'Median_v_dut'], axis=1)

df_i_arc = df.iloc[:, :-500]
df1 = df.iloc[:, 250:]
df_i_dc = df1.iloc[:, :-250]
df_v_dut = df.iloc[:, 500:]

#====================================================
# Extra Trees Implementation
#====================================================
X = pd.concat([df_i_dc, df_v_dut], axis=1)
y = input_data['Target']
groups = input_data['Group'].values

# Set a fixed seed for reproducibility
fixed_seed = 42

# Initialize LeaveOneGroupOut
logo = LeaveOneGroupOut()

# Range of threshold from 0 to 1 with a step-size increment of 0.01
thresholds = np.arange(0, 1, 0.01)
# To store the scores and thresholds
all_y_true = []
all_y_prob = []

# Perform Leave-One-Group-Out cross-validation
for fold_i, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    random_state = fixed_seed + fold_i
    # change hyperaparemeters to default for Inverter 1 and AutoGluon for Inverter 2
    et_classifier = ExtraTreesClassifier(
        n_estimators=100,
        #max_leaf_nodes=15000,
        random_state=random_state,
        n_jobs=-1
    )
    et_classifier.fit(X_train, y_train)
    y_prob = et_classifier.predict_proba(X_test)[:, 1]
    
    all_y_true.extend(y_test)
    all_y_prob.extend(y_prob)

all_y_true = np.array(all_y_true)
all_y_prob = np.array(all_y_prob)


#====================================================
# Metric Calculations
#====================================================
# Calculate metrics for all thresholds
accuracies, fnrs, fprs = calculate_metrics(all_y_true, all_y_prob, thresholds)

# Calculate ROC AUC
roc_auc = roc_auc_score(all_y_true, all_y_prob)
print(f'ROC AUC: {roc_auc:.2f}')

# Compute the weighted metric and find the best threshold
weighted_metric = [0.6 * fnr + 0.4 * fpr for fnr, fpr in zip(fnrs, fprs)]
best_weighted_index = np.argmin(weighted_metric)
best_weighted_threshold = thresholds[best_weighted_index]

# Print the results
print(f'Best Threshold (Minimizing 0.6*FNR + 0.4*FPR): {best_weighted_threshold:.2f}')
print(f'FNR at Best Threshold: {fnrs[best_weighted_index]:.4f}')
print(f'FPR at Best Threshold: {fprs[best_weighted_index]:.4f}')
print(f'Accuracy at Best Threshold: {accuracies[best_weighted_index]:.4f}')

# To plot in percentages
fnrs_percent = [value * 100 for value in fnrs]
fprs_percent = [value * 100 for value in fprs]

# Plot FNR and FPR trade-off
plt.figure(figsize=(10, 6))
plt.plot(thresholds, fnrs_percent, label='FNR')
plt.plot(thresholds, fprs_percent, label='FPR')
plt.axvline(best_weighted_threshold, color='red', linestyle='--', label=f'Best Threshold ({best_weighted_threshold:.2f})')
plt.xlabel('Classification Threshold',fontsize=14)
plt.ylabel('Metric Score [%]',fontsize=14)
plt.title('Inverter 1',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.legend(fontsize=12)
plt.grid(which='both', linestyle='--', linewidth=0.5, color='grey')
plt.savefig("Tradeoff_I1.svg", format='svg', dpi=300)
plt.show()

