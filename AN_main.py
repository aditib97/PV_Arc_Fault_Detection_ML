# -*- coding: utf-8 -*-
"""
Adaptive Normalisation based time-series approach (Manual)

@author: Aditi Bongale
"""

import os
import pandas as pd
import time

from basic.model_validation import validate_model_LOGO_I1, validate_model_LOGO_I2
from basic.visualisation import plot_results
#====================================================
# Read in data
#====================================================
# Paths within directory
combined_dir = 'data/03_CombinedProcessedData_csv'
combined_csv_name = 'Combined_I1_4K_with_gapfilling.csv'

results_dir = "results"
results_csv_name = 'CVResults_I1_4K_with_gapfilling_AN.csv'

# Read the CSV file into a DataFrame
input_data = pd.read_csv(os.path.join(combined_dir, combined_csv_name))
print('Tabular data read')

#====================================================
# Data processing - With Normalisation
#====================================================
start_n = time.time()
# Seperate 3 inputs with their respective medians
df = input_data.drop(['Target','Group','Median_i_dc'], axis=1)
df_i_arc = df.iloc[:, :-501]
df1 = df.iloc[:, 250:]
df_i_dc = pd.concat([df1.iloc[:, :-251], input_data['Median_i_dc']], axis=1)
df_v_dut = df.iloc[:, 500:]

# Normalisation
df_v_dut_norm = df_v_dut.copy()
df_i_dc_norm = df_i_dc.copy()

# Iterate through each row and divide by the 'Median' value
for index, row in df_i_dc.iterrows():
    df_i_dc_norm.loc[index, df_i_dc_norm.columns != 'Median_i_dc'] = row[df_i_dc.columns != 'Median_i_dc'] / row['Median_i_dc']

# Iterate through each row and divide by the 'Median' value
for index, row in df_v_dut_norm.iterrows():
    df_v_dut_norm.loc[index, df_v_dut_norm.columns != 'Median_v_dut'] = row[df_v_dut.columns != 'Median_v_dut'] / row['Median_v_dut']

df_i_dc_norm = df_i_dc_norm.drop(['Median_i_dc'], axis=1)
df_v_dut_norm = df_v_dut_norm.drop(['Median_v_dut'], axis=1)

end_n = time.time()
process = end_n - start_n
#====================================================
# LOGO-EXTRA TREES 
#====================================================
# Prepare data for model cross validation
X = pd.concat([df_i_dc_norm, df_v_dut_norm], axis=1)
y = input_data['Target']
groups = input_data['Group'].values 

# Train model with Leave-One-Group-Out Cross Validation
# For Inverter 1
results_df, y_preds, y_tests = validate_model_LOGO_I1(X, y, groups, mixed=30)
# For Inverter 2
# results_df, y_preds, y_tests = validate_model_LOGO_I2(X, y, groups, mixed=19)            

# Save the results to a CSV file
results_filepath = os.path.join(results_dir, results_csv_name)
results_df.to_csv(results_filepath, index=False, sep=',', encoding='utf-8')

# Visualize the results as performance graph
plot_results(y_preds, y_tests, results_dir)