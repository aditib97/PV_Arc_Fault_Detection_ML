# -*- coding: utf-8 -*-
"""
Wavelet Packet Entropy based approach

@author: Aditi Bongale
"""

import os
import numpy as np
import pandas as pd
import time
import pywt

from basic.model_validation import validate_model_LOGO_I1, validate_model_LOGO_I2

#====================================================
# Function for WPE
#====================================================
def calculate_wavelet_packet_entropy(signal, level=3, wavelet='db9'):
    
    # Perform wavelet packet decomposition
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=level)
    nodes = wp.get_level(level, order="freq")  # Get nodes sorted by frequency
    coeffs = [node.data for node in nodes]     # Extract coefficients

    # Compute entropy for each node
    entropy = np.array([
        -np.sum(c ** 2) * np.log(np.sum(c ** 2)) if np.sum(c ** 2) > 0 else 0 
        for c in coeffs
    ])
    return entropy

#====================================================
# Read in data
#====================================================
# Paths within directory
combined_dir = 'data/03_CombinedProcessedData_csv'
combined_csv_name = 'Combined_I1_32K_with_gapfilling.csv'

results_dir = "results"
results_csv_name = 'CVResults_I1_32K_WPE_gapfilling.csv'

# Read the CSV file into a DataFrame
input_data = pd.read_csv(os.path.join(combined_dir, combined_csv_name))
print('Tabular data read')

#====================================================
# Data Processing
#====================================================

start_n = time.time()
df_i_arc = input_data.drop(['Target', 'Group'], axis=1)
# Calculate wavelet packet entropy for each row
entropy_vectors = []
for index, row in df_i_arc.iterrows():
    # Extract the row values as a signal
    original_signal = row.values
    
    # Compute wavelet packet entropy
    entropy = calculate_wavelet_packet_entropy(original_signal, level=3, wavelet='db9')
    entropy_vectors.append(entropy)

# Create a DataFrame of entropy vectors
df_entropy = pd.DataFrame(entropy_vectors)

end_n = time.time()
process = end_n - start_n

#====================================================
# LOGO-EXTRA TREES 
#====================================================
# Prepare data for model cross validation
X = df_entropy
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