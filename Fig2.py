"""
Author: Sulu
Python version: 3.9
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from matplotlib.lines import Line2D  # For custom legend handles
import matplotlib.pylab as pylab
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Set plot parameters
params = {'legend.fontsize': 'x-large', 'figure.figsize': (15, 20), 'axes.labelsize': 'x-large', 
          'axes.titlesize': 'x-large', 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)

hdf5filepath='Data_hdf5format_ver3/VC'
hdf5file_list=[]
for file in os.listdir(hdf5filepath):
    if file.endswith ("hdf5"):
        hdf5file_list.append(file)

dfs_allcells=[]
for file in hdf5file_list:
    df=pd.read_hdf(f"{hdf5filepath}/{file}")
    dfs_allcells.append(df)

def cell_physio(dfs_allcells):
    results=[]
    for df in dfs_allcells:
        # Filter for Exc and Inh conditions
        exc_df = df[df['clampPotential'] == -70]
        inh_df = df[df['clampPotential'] == 0]
        
        exc_mean_holdI = exc_df['holdI'].mean() if not exc_df.empty else float('nan')
        inh_mean_holdI = inh_df['holdI'].mean() if not inh_df.empty else float('nan')
        
        exc_mean_IR = exc_df['IR'].mean() if not exc_df.empty else float('nan')
        inh_mean_IR = inh_df['IR'].mean() if not inh_df.empty else float('nan')
        
        exc_mean_stpr = exc_df['stpr'].mean() if not exc_df.empty else float('nan')
        inh_mean_stpr = inh_df['stpr'].mean() if not inh_df.empty else float('nan')
        
        # Store the results
        results.append({
            'holdI_exc': exc_mean_holdI,
            'holdI_inh': inh_mean_holdI,
            'IR_exc': exc_mean_IR,
            'IR_inh': inh_mean_IR,
            'stpr_exc': exc_mean_stpr,
            'stpr_inh': inh_mean_stpr
        })
    results_df = pd.DataFrame(results)
    return results_df
    
def violinplot_cellparams():
    results_df=cell_physio(dfs_allcells)
    num_cols = len(results_df.columns)
    plt.figure(figsize=(10, 3))
    colors = ['blue', 'red']
    
    for i, column in enumerate(results_df.columns):
        color = colors[i % len(colors)]
        
        plt.subplot(1, num_cols, i + 1)
        sns.violinplot(y=results_df[column], inner="point", color=color, alpha=0.5)

        mean_value = results_df[column].mean()
        plt.scatter(
            0.0, 
            mean_value, 
            color='black', 
            marker='_', 
            s=1000, 
            label='Mean'
        )
        # plt.title(f'{column}')
        plt.xlabel(column)
        plt.gca().set_ylabel('')
    plt.tight_layout()
    plt.show()

def time_series(dfs_allcells):
    plt.figure(figsize=(15, 10))

    plot_elements = []
    labels = []
    
    for df in dfs_allcells:
        exc_df = df[df['clampPotential'] == -70]
        inh_df = df[df['clampPotential'] == 0]
        
        # Plot and store elements for excitation
        exc_holdI, = plt.plot(exc_df['holdI'], color='blue')
        exc_IR, = plt.plot(exc_df['IR'], color='cyan')
        exc_stpr, = plt.plot(exc_df['stpr'], color='black')
        
        # Plot and store elements for inhibition
        inh_holdI, = plt.plot(inh_df['holdI'], color='red')
        inh_IR, = plt.plot(inh_df['IR'], color='orange')
        inh_stpr, = plt.plot(inh_df['stpr'], color='salmon')

    plot_elements.extend([exc_holdI, exc_IR, exc_stpr, inh_holdI, inh_IR, inh_stpr])
    labels.extend(['Exc - holdI', 'Exc - IR', 'Exc - stpr', 'Inh - holdI', 'Inh - IR', 'Inh - stpr'])
    
    # Set labels and legend
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(plot_elements, labels)
    
    plt.tight_layout()
    plt.ylim(-500, 1000)
    plt.show()


time_series(dfs_allcells)
