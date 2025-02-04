# -*- coding: utf-8 -*-
"""
Adaptive Normalisation: Update factor based on standard deviation 
and time. (Automatic)

@author: Aditi Bongale
"""

import os
import pandas as pd
import time
import matplotlib.pyplot as plt

from basic.model_validation import validate_model_LOGO_I1, validate_model_LOGO_I2
from basic.visualisation import plot_results

#====================================================
# Read in data
#====================================================
# Paths within directory
combined_dir = 'data/03_CombinedProcessedData_csv'
combined_csv_name = 'Combined_I2_4K_with_gapfilling.csv'

# Read the CSV file into a DataFrame
input_data = pd.read_csv(os.path.join(combined_dir, combined_csv_name))
print('Tabular data read')

#====================================================
# Data processing - With Normalisation
#====================================================
start_n = time.time()

df = input_data.drop(['Target', 'Group'], axis=1)

# Calculate the std dev of the first 25 samples of each row
df['StdDev_First_25_idc'] = df.iloc[:, 250:275].std(axis=1)
df['StdDev_First_25_vdut'] = df.iloc[:, 500:525].std(axis=1)

# Calculate the median of the first 25 samples of each row
df['Median_First_25_idc'] = df.iloc[:, 250:275].median(axis=1)
df['Median_First_25_vdut'] = df.iloc[:, 500:525].median(axis=1)         

end_n = time.time()
middle = end_n - start_n

# Initialize the new column with NaN values
df['Norm_v'] = pd.NA
df['Norm_i'] = pd.NA

rowcount = 0

# Set the first value of Norm to the first value of Median_First_25
df.at[0, 'Norm_v'] = df.at[0, 'Median_First_25_vdut']
df.at[0, 'Norm_i'] = df.at[0, 'Median_First_25_idc']

# Iterate through the DataFrame for current
for i in range(1, len(df)):
    condition_1 = rowcount == 80 # 80 is for 500ms
    condition_2 = df.at[i, 'StdDev_First_25_idc'] > 1
    if condition_1 and condition_2:
        rowcount = 0
        df.at[i, 'Norm_i'] = df.at[i-1, 'Norm_i']       
    elif condition_1 and not condition_2:
        rowcount = 0
        df.at[i, 'Norm_i'] = df.at[i, 'Median_First_25_idc']
    elif not condition_1 and condition_2:
        rowcount = 0
        df.at[i, 'Norm_i'] = df.at[i-1, 'Norm_i']
    else:
        rowcount = rowcount + 1
        df.at[i, 'Norm_i'] = df.at[i-1, 'Norm_i']   

rowcount = 0
# Iterate through the DataFrame for voltsge
for i in range(1, len(df)):
    condition_1 = rowcount == 80
    condition_2 = df.at[i, 'StdDev_First_25_vdut'] > 1
    if condition_1 and condition_2:
        rowcount = 0
        df.at[i, 'Norm_v'] = df.at[i-1, 'Norm_v']       
    elif condition_1 and not condition_2:
        rowcount = 0
        df.at[i, 'Norm_v'] = df.at[i, 'Median_First_25_vdut']
    elif not condition_1 and condition_2:
        rowcount = 0
        df.at[i, 'Norm_v'] = df.at[i-1, 'Norm_v']
    else:
        rowcount = rowcount + 1
        df.at[i, 'Norm_v'] = df.at[i-1, 'Norm_v']   

# Plot for visualisations
plt.figure(figsize=(10, 6))

# Primary y-axis
ax1 = plt.gca()
line1, = ax1.plot(df['Norm_i'][:2524], label='Automatic median', color='orange')
line2, = ax1.plot(df['Median_i_dc'][:2524], label='Manual median', color='green')
ax1.set_xlabel('Samples')
ax1.set_ylabel('idc values [A]')
ax1.set_title('StdDev_First_25_idc <= 1 && rowcount == 80')

# Secondary y-axis
ax2 = ax1.twinx()
line3, = ax2.plot(df['StdDev_First_25_idc'][:2524], label='STD', color='blue')
ax2.axhline(y=1, color='red', linestyle='dotted', linewidth=1)
ax2.set_ylabel('StdDev_First_25_idc')

# Combine legends
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
lines_2, labels_2 = ax2.get_legend_handles_labels()
plt.legend(lines, labels, loc='upper right')
plt.show()


plt.figure(figsize=(10, 6))

# Primary y-axis
ax1 = plt.gca()
line1, = ax1.plot(df['Norm_v'][:2524], label='Automatic median', color='orange')
line2, = ax1.plot(df['Median_v_dut'][:2524], label='Manual median', color='green')
ax1.set_xlabel('Samples')
ax1.set_ylabel('vdut values [V]')
ax1.set_title('StdDev_First_25_vdut <= 1 && rowcount == 80')

# Secondary y-axis
ax2 = ax1.twinx()
line3, = ax2.plot(df['StdDev_First_25_vdut'][:2524], label='STD', color='blue')
ax2.axhline(y=1, color='red', linestyle='dotted', linewidth=1)
ax2.set_ylabel('StdDev_First_25_vdut')

# Combine legends
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
lines_2, labels_2 = ax2.get_legend_handles_labels()
plt.legend(lines, labels, loc='upper right')
plt.show()



