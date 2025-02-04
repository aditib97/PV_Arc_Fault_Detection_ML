# -*- coding: utf-8 -*-
"""
PV Arc Fault Detection using Extra Trees

@author: Aditi Bongale
"""

import os
import pandas as pd
import time
import tracemalloc

from basic.data_processing import data_preprocess
from basic.model_validation import validate_model_LOGO_I1, validate_model_LOGO_I2
from basic.visualisation import plot_results

#====================================================
# Declare directories
#====================================================
# Paths within directory
raw_dir_1 = 'data/01_RawData/01_Inverter1_Iconst_mode/I1_Category_A'
raw_dir_2 = 'data/01_RawData/01_Inverter1_Iconst_mode/I1_Category_B'

processed_dir = 'data/02_ProcessedData_csv'
processed_folder = 'I1_processed_4K'

combined_dir = 'data/03_CombinedProcessedData_csv'
combined_csv_1 = 'Combined_I1_4K_A.csv'
combined_csv_2 = 'Combined_I1_4K_B.csv'
combined_csv_final = 'Combined_I1_4K.csv'

results_dir = "results"
results_csv_name = 'CVResults_I1_4K.csv'
process_csv_name = 'Preprocess_time_I1_4K.csv'

#=======================================================
# Preprocess individual files and combine them into csv
# Returns the time required to perform individual tasks
#=======================================================

# Sampling rate for downsampling
rate = 4000 # mention in samples

# Lists to store results from both function calls
all_creates = []
all_pres = []
all_downsamples = []

tracemalloc.start() # Start measuring memory usage

#First Call for mixed data
creates, pres, downsamples = data_preprocess(
    raw_dir_1, processed_dir, processed_folder, 
    combined_dir, combined_csv_1, rate, mixed=1
)

# Append results #1
all_creates.extend(creates)
all_pres.extend(pres)
all_downsamples.extend(downsamples)

#Second Call for only non-arcing data
creates, pres, downsamples = data_preprocess(
    raw_dir_2, processed_dir, processed_folder, 
    combined_dir, combined_csv_2, rate, mixed=0
)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 10**6} MB")
print(f"Peak memory usage: {peak / 10**6} MB")
tracemalloc.stop() # Stop tracing

# Append results #2
all_creates.extend(creates)
all_pres.extend(pres)
all_downsamples.extend(downsamples)

# Create a DataFrame to store the process times
times_df = pd.DataFrame({
    'Creation Time': all_creates,
    'PreProcess Time': all_pres,
    'Downsample Time': all_downsamples
})

# Calculate averages
creation_time_avg = times_df['Creation Time'].mean()
preprocess_time_avg = times_df['PreProcess Time'].mean()
downsample_avg = times_df['Downsample Time'].mean()

# Print the averages of preprocessing time for each measurement
print(f"Creation Time Average: {creation_time_avg:.4f}")
print(f"PreProcess Time Average: {preprocess_time_avg:.4f}")
print(f"Downsample Average: {downsample_avg:.4f}")

# Save the process time to a CSV file
results_filepath = os.path.join(results_dir, process_csv_name)
times_df.to_csv(results_filepath, index=False, sep=',', encoding='utf-8')

#====================================================
# Combine mixed and nominal datasets
#====================================================
# Read both CSV files
df1 = pd.read_csv(os.path.join(combined_dir, combined_csv_1))
df2 = pd.read_csv(os.path.join(combined_dir, combined_csv_2)) 

# Concatenate the DataFrames vertically (stack the rows)
df_combined = pd.concat([df1, df2])
df_combined = df_combined.reset_index(drop=True)

# Save to a new CSV file- Final Combined CSV
df_combined.to_csv(os.path.join(combined_dir, combined_csv_final), index=False)

#====================================================
# No Normalisation
#====================================================
start_n = time.time()
# Seperate 3 inputs
df = df_combined.drop(['Target', 'Group', 'Median_i_dc', 'Median_v_dut'], axis=1)
df_i_arc = df.iloc[:, :-500] # 4000 for 32 kS/s
df1 = df.iloc[:, 250:] # 2000 for 32 kS/s
df_i_dc = df1.iloc[:, :-250] # 2000 for 32 kS/s
df_v_dut = df.iloc[:, 500:] # 4000 for 32 kS/s

end_n = time.time()
process = end_n - start_n
#====================================================
# Extra Trees Classifier Implementation
#====================================================
# Prepare data for model cross validation
X = pd.concat([df_i_dc, df_v_dut], axis=1)             
y = df_combined['Target']
groups = df_combined['Group'].values

# Train model with Leave-One-Group-Out Cross Validation
# For Inverter 1
results_df, y_preds, y_tests = validate_model_LOGO_I1(X, y, groups, mixed=29)
# # For Inverter 2
# results_df, y_preds, y_tests = validate_model_LOGO_I2(X, y, groups, mixed=18)            

# Save the results to a CSV file
results_filepath = os.path.join(results_dir, results_csv_name)
results_df.to_csv(results_filepath, index=False, sep=',', encoding='utf-8')

# Visualize the results as performance graph
plot_results(y_preds, y_tests, results_dir)