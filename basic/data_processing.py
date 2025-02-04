# -*- coding: utf-8 -*-
"""
Data Processing Functions

@author: Aditi Bongale
"""

import os
import numpy as np
import pandas as pd
import gc
import time

from basic.data_creation import read_data, calculate_arc_start, downsample_data, create_target_column
from basic.visualisation import plot_iarc_data, plot_data

#====================================================
#Declaration of variables 
#====================================================

window = 62.5 #sliding window size in ms
step = 6.25   #window step size in ms

#===============================================================
#Function to calculate the parameters for tabular representation
#===============================================================

def calculate_parameters(rate, window, step):
    # Calculate sampling interval in ms
    sampling_interval = (1 / rate) * 1000  # in ms
    
    # Calculate columns for window size and step size 
    window_size = int(window / sampling_interval) 
    step_size = int(step / sampling_interval)     
    
    # Form target_freq as a string (in ms)
    target_freq = f"{sampling_interval}ms"
    
    return sampling_interval, window_size, step_size, target_freq

#====================================================
#Function calls for creating sliding windows
#====================================================

def create_sliding_windows(column_data, window_size, step_size):

    column_data = column_data.values if isinstance(column_data, pd.Series) else column_data
    num_windows = (len(column_data) - window_size) // step_size + 1
    
    # Initialize an empty list to store the windows
    windows = []
    
    # Loop through the data using the sliding window approach
    for i in range(0, num_windows * step_size, step_size):
        window = column_data[i:i + window_size]
        if len(window) == window_size:  # Ensure the window has the correct size
            windows.append(window)
    
    # Convert the list of windows to a numpy array
    windows_array = np.array(windows)
    return windows_array    

# Generate names for the columns with interval time
def generate_column_names(feature_name, window_size, interval):
    return [f"{feature_name}_{interval * (i+1):.2f}" for i in range(window_size)]

#==========================================================
#Create Tabular data - Preprocess each file and add features
#===========================================================

def data_preprocess(raw_dir, processed_dir, processed_folder, combined_dir, combined_csv, rate, mixed):
    # Calculate parameters
    sampling_interval, window_size, step_size, target_freq = calculate_parameters(rate, window, step)
    # List to hold individual DataFrames
    df_list = []
    # To store the scores for each split
    creates, pres, downsamples = [], [], []
    
    # Process each file
    for root, dirs, files in os.walk(raw_dir):
        for filename in files:
            if filename.endswith('.d7d'):
                file_path = os.path.join(root, filename)
                # Create a similar folder structure for saving outputs
                relative_path = os.path.relpath(root, raw_dir)
                output_dir = os.path.join(processed_dir, processed_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                #====================================================
                #Data Creation and Data Pre-processing
                #====================================================                              
                start_c = time.time()
                # Read in data
                data = read_data(file_path)
                Zeit = data.index
                
                # Calculate sampling rate
                SR = round(1 / (Zeit[2] - Zeit[1]))
                
                # Calculate start of arc
                arc_starts = calculate_arc_start(data, SR)
                # Downsampling
                start_d = time.time()
                df_resampled = downsample_data(data, target_freq)
                end_d = time.time()
    
                # Create the target column
                target = create_target_column(df_resampled, arc_starts, Zeit, mixed)
                
                end_c = time.time()
                create = end_c - start_c
                downsample = end_d - start_d
                downsamples.append(downsample)
                creates.append(create)
                
                # Sliding Window
                start_p = time.time()
                
                # Create sliding windows for each column by passing the specific column
                i_arc_windows = create_sliding_windows(df_resampled['i_arc'], window_size, step_size)
                i_dc_windows = create_sliding_windows(df_resampled['i_dc'], window_size, step_size)
                v_dut_windows = create_sliding_windows(df_resampled['v_dut'], window_size, step_size)
                # Combine the arrays across columns
                combined_windows_array = np.hstack((i_arc_windows, i_dc_windows, v_dut_windows))
                # Generate column names for each feature
                i_arc_column_names = generate_column_names('i_arc', window_size, sampling_interval)
                i_dc_column_names = generate_column_names('i_dc', window_size, sampling_interval)
                v_dut_column_names = generate_column_names('v_dut', window_size, sampling_interval)
                # Combine all column names
                all_column_names = i_arc_column_names + i_dc_column_names + v_dut_column_names
    
                # Sliding window for the target column
                target_windows = create_sliding_windows(target, window_size, step_size)
                # Iterate through each row and check for the presence of a 1
                target_array = np.zeros((target_windows.shape[0], 1))
                for i in range(target_windows.shape[0]):
                    if 1 in target_windows[i]:
                        target_array[i] = 1
                
                # Create Tabular data
                tabular_data = pd.DataFrame(combined_windows_array, columns=all_column_names)
                tabular_data['Target'] = target_array
                
                # Extract the Group/subfolder name
                tabular_data['Group'] = os.path.basename(root)
                
                # Extract the median of measurement with first 25 samples for 4k sampled data
                tabular_data['Median_i_dc'] = np.median(i_dc_windows[0, 0:step_size])
                tabular_data['Median_v_dut'] = np.median(v_dut_windows[0, 0:step_size])
                
                end_p = time.time()
                pre = end_p - start_p
                pres.append(pre)
                #====================================================
                #Save Tabular Data as csv and plot for visualisation
                #====================================================                
                csv_file_name = filename.replace('.d7d', '_4k_25.csv')
                tabular_data.to_csv(os.path.join(output_dir, csv_file_name), index=False)
                
                plot_iarc_data(df_resampled['i_arc'].values, target, filename, output_dir)
                plot_data(df_resampled['i_dc'].values, df_resampled['v_dut'].values, target, filename, output_dir, rate)

                # Combine all csv
                df_list.append(tabular_data)
                #====================================================
                # Free up Memory
                #====================================================
                # Delete the variables to free up memory
                del combined_windows_array, i_arc_windows, v_dut_windows, i_dc_windows, df_resampled 
                del tabular_data
                gc.collect()  # Run garbage collector
                
                print('Done')
                
    print('Tabular data creation done for all files')
    # Concatenate all DataFrames in the list into a single DataFrame
    df_combined = pd.concat(df_list, ignore_index=True)
    print(f"Combined tabular data shape: {df_combined.shape}")

    # Save the combined DataFrame to a new CSV file
    output_path = os.path.join(combined_dir, combined_csv)
    df_combined.to_csv(output_path, index=False)

    # Print the first few rows of the combined DataFrame
    print(df_combined.head())

    return creates, pres, downsamples
    

