# -*- coding: utf-8 -*-
"""
Functions to help Data Creation

@author: Aditi Bongale
"""
import numpy as np
import pandas as pd
import dwdatareader as dw

#====================================================
#Function calls for reading d7d data from Dewesoft
#====================================================

def d7d_read_column_names(filename):
    with dw.open(filename) as f:
        df_columns = []
        for ch in f.values():
            df_columns.append(ch.name)
        return df_columns

def d7d_load_columns(filename, column_names):
    with dw.open(filename) as f:
        df = pd.DataFrame()
        for index, col in enumerate(column_names):
            df[col] = f[col].series()
        return df

def read_data(file_path):
    columns = d7d_read_column_names(file_path)
    data = d7d_load_columns(file_path, columns)
    return data

#====================================================
#Calculation of start of arc 
#====================================================

def calculate_arc_start(data, SR):
    
    # Minimum consecutive time where conditions have to be 
    # met to confirm arc start and stop [in s]
    min_time = 1e-3 #1ms
    Zeit = data.index
    v_arc = data['V_Arc']
    
    # Number of consecutive datapoints to confirm arcing start/stop
    n_seq = round(min_time * SR)
    
    # Check where threshold voltage for arc ignition is reached (1=True, 0=False)
    v_threshold_series = v_arc > 10
    v_threshold_series = v_threshold_series.astype(int)

    # Compute consecutive count of voltage threshold breaks
    consecutive_v_threshold = v_threshold_series.groupby((v_threshold_series != v_threshold_series.shift()).cumsum()).cumsum()

    # Define arc start index when voltage threshold is broken for the defined number of consecutive datapoints
    if any(consecutive_v_threshold == n_seq):
        arc_start_time = consecutive_v_threshold.index[consecutive_v_threshold.tolist().index(n_seq)]
        arc_starts = Zeit.tolist().index(arc_start_time)
        # Move index to first threshold break
        arc_starts -= n_seq - 1
    else:
        arc_starts = 0
        
    return arc_starts
    
#====================================================
#Downsample data to target frequency
#====================================================

def downsample_data(data, target_freq):
    
    # Convert data to NumPy array
    total_i_arc = data['I_Arc'].values
    total_v_dut = data['V_DUT'].values
    total_i_dc = data['I_DC'].values
    
    # Create the TimedeltaIndex with 2 microsecond intervals
    num_periods = len(total_i_arc)
    times = pd.timedelta_range(start='0us', periods=num_periods, freq='2U')
    
    # Create a DataFrame and update its index with the 'times'
    df_4s = pd.DataFrame({'i_arc': total_i_arc, 'i_dc': total_i_dc, 'v_dut': total_v_dut}, index=times)
    
    # Resample to 4KHz (1/4KHz = 0.25ms)
    df_4s_resampled = df_4s.resample(target_freq).mean()
    
    return df_4s_resampled

#====================================================
#Create the target column based on the arc start time
#====================================================

def create_target_column(data_resampled, arc_start, original_time_index, mixed):

    data_length = len(data_resampled['i_arc'])
    # Calculate arc start for resampled data
    arc_starts_resampled = round((arc_start * data_length) / len(original_time_index))
    
    # Initialize target column (0 if no arc detected, 1 if arc detected)
    target = np.zeros(data_length)
    # Plot ones only if its a mixed scenario, nameley Category A
    if mixed == 1:
        target[arc_starts_resampled:] = 1
    
    return target 

