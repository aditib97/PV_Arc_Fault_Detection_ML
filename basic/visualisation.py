# -*- coding: utf-8 -*-
"""
Functions For Visualisations and saving predictions

@author: Aditi Bongale
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import numpy as np

#====================================================
#Function calls for plotting i_arc vs target
#====================================================
def plot_iarc_data(i_arc_4k, target, filename, o_dir):   
    fig, ax1 = plt.subplots(figsize=(9, 4))
    # Plot voltage on primary y-axis
    ax1.plot(i_arc_4k, label="i_arc", alpha=0.8)
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Current [A]")
    ax1.grid(which='both', linestyle='--', linewidth=0.5, color='grey')
    # Adding arc current on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(target, label="target", color="red",linestyle='--', alpha=0.8)
    ax2.set_ylabel("Binary Label")
    ax2.set_yticks([0, 1])
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title("Input signals sampled at 4 kS/s")
    plt.tight_layout()
    #plt.savefig(os.path.join(o_dir, filename+'_iarc_4k.png'))
    
    
#====================================================
#Function calls for plotting i_dc, v_dut vs target
#====================================================    
def plot_data(i_dc_4k, v_dut_4k, target, filename, o_dir, rate):

    fig, ax1 = plt.subplots(figsize=(9, 4))
    
    time_interval = 1 / rate
    new_index = np.arange(0, len(i_dc_4k) * time_interval, time_interval)[:len(i_dc_4k)] 

    i_dc_4k = pd.Series(i_dc_4k, index=new_index)
    v_dut_4k = pd.Series(v_dut_4k, index=new_index)
    
    # Plot voltage on primary y-axis
    ax1.plot(new_index, v_dut_4k, label="v_dut", color="blue", alpha=0.8)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("v_dut Voltage [V]")
    ax1.grid(which='both', linestyle='--', linewidth=0.5, color='grey')
    # Adding arc current on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(new_index, i_dc_4k, label="i_dc", color="orange", alpha=0.8)
    ax2.set_ylabel("i_dc Current [A]")   
    ax3 = ax1.twinx()
    ax3.plot(new_index, target, label="target", color="red",linestyle='--', alpha=0.8)
    ax3.spines['right'].set_position(('outward', 40))  
    ax3.set_ylabel("Binary Label")
    ax3.set_yticks([0, 1])
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title("Input signals sampled at 4 kS/s")
    plt.tight_layout()
    #plt.savefig("I1_1_waveform_poor.svg", format='svg', dpi=300)
    plt.show()
    #plt.savefig(os.path.join(o_dir, filename+'_idc_vdut_4k.png'))
    
#====================================================
#Function calls for plotting performance graphs
#====================================================
def plot_results(y_preds, y_tests, o_dir):
    """
    Plots the predictions against the ground truths.
    """
    for i, (array1, array2) in enumerate(zip(y_preds, y_tests)):
        plt.figure(figsize=(9, 4))
        
        df1 = pd.DataFrame(array1).reset_index(drop=True)
        df2 = pd.DataFrame(array2).reset_index(drop=True)
        
        plt.plot(df1, label='y_pred', color='red', alpha=0.8)
        plt.plot(df2, label='y_test',alpha=0.8)

        plt.xlabel('Index')
        plt.ylabel('Binary Label')
        plt.title(f'Plot of Group {i+1} pred vs ground truth')
        plt.legend()
        plt.grid()
        plt.yticks([0, 1])
        plt.show()
        #plt.savefig(os.path.join(o_dir, f'Group {i+1}.png'))

#====================================================
#Function calls for saving prediction and ground truth
#====================================================     
def save_results(y_preds, y_tests, o_dir):
    for i, (array1, array2) in enumerate(zip(y_preds, y_tests)):

        # df1 = pd.DataFrame(array1).reset_index(drop=True)
        # df2 = pd.DataFrame(array2).reset_index(drop=True)

        data_path = os.path.join(o_dir, f'Group_{i+1}_data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump({"y_pred": array1, "y_test": array2}, f)
            