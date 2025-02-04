# -*- coding: utf-8 -*-
"""
Peformance calculation after adding additional time-domain features.

@author: Aditi Bongale
"""
import os
import numpy as np
import pandas as pd
import time

from basic.model_validation import validate_model_LOGO_I1, validate_model_LOGO_I2
#====================================================
# Function to calculate additional features
#====================================================
def calculate_features(row):
    # Convert the row to a NumPy array for easier mathematical operations
    row = np.array(row)

    mu = np.mean(row)   # Calculate mean (μ)
    M = np.median(row)  # Calculate median (M)
    V = np.var(row)  # Calculate variance (V)
    STD = np.std(row)  # Calculate standard deviation (STD)
    Diff = np.max(row) - np.min(row) # Calculate difference between max and min (diff)
    RMS = np.sqrt(np.mean(row ** 2)) # Calculate RMS
    
    return mu, M, V, STD, Diff, RMS

#====================================================
# Read in data
#====================================================
# Paths within directory
combined_dir = 'data/03_CombinedProcessedData_csv'
combined_csv_name = 'Combined_I2_4K.csv'

results_dir = "results"
results_csv_name = 'CVResults_I2_4K_features.csv'

# Read the CSV file into a DataFrame
input_data = pd.read_csv(os.path.join(combined_dir, combined_csv_name))
print('Tabular data read')

#====================================================
# Processing - No Normalisation + Features
#====================================================
start_n = time.time()
# Seperate 3 inputs
df = input_data.drop(['Target', 'Group', 'Median_i_dc', 'Median_v_dut'], axis=1)
df1 = df.iloc[:, 250:] 
df_i_dc = df1.iloc[:, :-250] 
df_v_dut = df.iloc[:, 500:] 

results = df_i_dc.apply(calculate_features, axis=1)
results_df_idc = pd.DataFrame(results.tolist(), columns=['Mean_idc', 'Median_idc', 'Variance_idc', 'Std_idc', 'Diff_idc','RMS_idc'])

results = df_v_dut.apply(calculate_features, axis=1)
results_df_v_dut = pd.DataFrame(results.tolist(), columns=['Mean_vdut', 'Median_vdut', 'Variance_vdut', 'Std_vdut', 'Diff_vdut','RMS_vdut'])

end_n = time.time()
process = end_n - start_n

#====================================================
# Extra Trees Classifier Implementation
#====================================================
X = pd.concat([df_i_dc, df_v_dut, results_df_idc, results_df_v_dut], axis=1)
y = input_data['Target']
groups = input_data['Group'].values

# Train model with Leave-One-Group-Out Cross Validation
# For Inverter 1
#results_df, y_preds, y_tests = validate_model_LOGO_I1(X, y, groups, mixed=29)
# # For Inverter 2, set to I1 for using default hyperparameters. 
results_df, y_preds, y_tests = validate_model_LOGO_I1(X, y, groups, mixed=18)            

# Save the results to a CSV file
results_filepath = os.path.join(results_dir, results_csv_name)
results_df.to_csv(results_filepath, index=False, sep=',', encoding='utf-8')
