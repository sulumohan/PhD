"""
Author: Sulu
Python version: 3.9
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import statsmodels.api as sm
import pylab
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.gridspec as gridspec

# Set plot parameters
params = {'legend.fontsize': 'x-large', 'figure.figsize': (15, 20), 'axes.labelsize': 'x-large', 
          'axes.titlesize': 'x-large', 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)

hdf5filepath = 'Data_hdf5format_ver3/VC'
hdf5file_list = [file for file in os.listdir(hdf5filepath) if file.endswith("hdf5")]

dfs_allcells = [pd.read_hdf(f"{hdf5filepath}/{file}") for file in hdf5file_list]

def cell_physio(dfs_allcells):
    results = []
    for df in dfs_allcells:
        exc_df = df[df['clampPotential'] == -70]
        inh_df = df[df['clampPotential'] == 0]
        
        exc_mean_holdI = exc_df['holdI'].mean() if not exc_df.empty else float('nan')
        inh_mean_holdI = inh_df['holdI'].mean() if not inh_df.empty else float('nan')
        
        exc_mean_IR = exc_df['IR'].mean() if not exc_df.empty else float('nan')
        inh_mean_IR = inh_df['IR'].mean() if not inh_df.empty else float('nan')
        
        exc_mean_stpr = exc_df['stpr'].mean() if not exc_df.empty else float('nan')
        inh_mean_stpr = inh_df['stpr'].mean() if not inh_df.empty else float('nan')
        
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

def violinplot_cellparams(ax):
    results_df = cell_physio(dfs_allcells)
    num_cols = len(results_df.columns)
    colors = ['blue', 'red']
    
    for i, column in enumerate(results_df.columns):
        color = colors[i % len(colors)]
        sns.violinplot(y=results_df[column], inner="point", color=color, alpha=0.5, ax=ax[i])
        
        mean_value = results_df[column].mean()
        ax[i].scatter(
            0.0, 
            mean_value, 
            color='black', 
            marker='_', 
            s=1000, 
            label='Mean'
        )
        ax[i].set_xlabel(column)
        ax[i].set_ylabel('')
        ax[i].legend()
        
        # Make spines invisible but keep axis lines
        for spine in ax[i].spines.values():
            spine.set_visible(False)
        ax[i].get_xaxis().set_ticks_position('none')  # Remove x-ticks but keep axis line
        ax[i].get_yaxis().set_ticks_position('none')  # Remove y-ticks but keep axis line

def time_series(ax):
    plot_elements = []
    labels = []
    
    for df in dfs_allcells:
        exc_df = df[df['clampPotential'] == -70]
        inh_df = df[df['clampPotential'] == 0]
        
        # Plot and store elements for excitation
        exc_holdI, = ax.plot(exc_df['holdI'], color='blue')
        exc_IR, = ax.plot(exc_df['IR'], color='cyan')
        exc_stpr, = ax.plot(exc_df['stpr'], color='black')
        
        # Plot and store elements for inhibition
        inh_holdI, = ax.plot(inh_df['holdI'], color='red')
        inh_IR, = ax.plot(inh_df['IR'], color='orange')
        inh_stpr, = ax.plot(inh_df['stpr'], color='salmon')

    plot_elements.extend([exc_holdI, exc_IR, exc_stpr, inh_holdI, inh_IR, inh_stpr])
    labels.extend(['Exc - holdI', 'Exc - IR', 'Exc - stpr', 'Inh - holdI', 'Inh - IR', 'Inh - stpr'])
    
    # Set labels and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend(plot_elements, labels)
    
    ax.set_ylim(-500, 1000)
    
    # Make spines invisible but keep axis lines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.get_xaxis().set_ticks_position('none')  # Remove x-ticks but keep axis line
    ax.get_yaxis().set_ticks_position('none')  # Remove y-ticks but keep axis line

def plot_scatter_with_regression_and_r2(ax):
    r2_stpr_holdI = []
    r2_stpr_IR = []
    
    for df in dfs_allcells:
        if {'stpr', 'holdI'}.issubset(df.columns):
            X = sm.add_constant(df['stpr'])
            model = sm.OLS(df['holdI'], X).fit()
            r2_stpr_holdI.append(model.rsquared)
        
        if {'stpr', 'IR'}.issubset(df.columns):
            X = sm.add_constant(df['stpr'])
            model = sm.OLS(df['IR'], X).fit()
            r2_stpr_IR.append(model.rsquared)

    r2_df = pd.DataFrame({
        'stpr vs holdI': r2_stpr_holdI,
        'stpr vs IR': r2_stpr_IR
    })

    sns.boxplot(data=r2_df, ax=ax)
    ax.set_ylabel('R² Value')
    
    # Make spines invisible but keep axis lines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.get_xaxis().set_ticks_position('none')  # Remove x-ticks but keep axis line
    ax.get_yaxis().set_ticks_position('none')  # Remove y-ticks but keep axis line

def plot_trial_avg_stpr(ax):
    legend_handles = []
    legend_labels = []
    
    for idx, numSQ in enumerate([1, 5, 15]):
        ax[idx].set_title(f'{numSQ} sq patterns', fontsize=20)
        ax[idx].set_xlabel('Pattern ID', fontsize=20)
        ax[idx].set_ylim(0, 5)
        ax[idx].tick_params(axis='both', which='major', labelsize=20)
    
        for i, df in enumerate(dfs_allcells):
            filtered_df = df[(df['clampPotential'] == -70) & (df['numSQ'] == numSQ)]
            grouped_data = filtered_df.groupby('numPattern')['stpr'].mean().reset_index()
    
            x = grouped_data['numPattern']
            y = grouped_data['stpr']
            color_for_df = plt.cm.tab10(i / len(dfs_allcells))
    
            scatter = ax[idx].scatter(x, y, color=color_for_df, label=f'Cell {i+1}', alpha=0.7, s=150)
    
            if idx == 0:
                legend_handles.append(scatter)
                legend_labels.append(f'Cell {i+1}')
                
        ax[idx].axhline(y=1, color='gray', linestyle='--', linewidth=2, label='STPR=1.0')
    
        if numSQ == 1:
            ax[idx].set_xlim(0.5, 15.5)
            ax[idx].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        elif numSQ == 5:
            ax[idx].set_xlim(0.5, 3.5)
            ax[idx].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        elif numSQ == 15:
            ax[idx].set_xlim(0.5, 1.5)
            ax[idx].set_xticks([1])
    
    # legend = ax[0].legend(legend_handles, legend_labels, title='Colors represent cells', bbox_to_anchor=(0.5, -0.08), loc='upper center', fontsize=15, ncol=len(dfs_allcells), frameon=False, handlelength=0.7, handletextpad=0.3)
    # legend.get_title().set_fontsize('15')
    
    # Make spines invisible but keep axis lines
    for subplot in ax:
        for spine in subplot.spines.values():
            spine.set_visible(False)
        subplot.get_xaxis().set_ticks_position('none')  # Remove x-ticks but keep axis line
        subplot.get_yaxis().set_ticks_position('none')  # Remove y-ticks but keep axis line

def exemplar_cells(ax):
    if len(ax) < 2:
        raise ValueError("Need at least 2 axes for the exemplar_cells function")
    
    for df_idx, ax_subplot in enumerate(ax):
        df = dfs_allcells[df_idx]
        filtered_df = df[(df['numSQ'] == 5) & (df['clampPotential'] == 0)]
        unique_combinations = filtered_df[['numPattern']].drop_duplicates()
        num_combinations = len(unique_combinations)
        colors = plt.cm.tab20(np.linspace(0, 1, num_combinations))
        
        for i, (numPattern, group) in enumerate(filtered_df.groupby('numPattern')):
            color = colors[i % num_combinations]
            ax_subplot.scatter(group['peakP1'], group['peakP2'], color=color, label=f'Pattern {numPattern}')
        
        ax_subplot.plot([filtered_df['peakP1'].min(), filtered_df['peakP1'].max()], 
                        [filtered_df['peakP1'].min(), filtered_df['peakP2'].max()], 
                        color='gray', linestyle='--', linewidth=2, label='x=y')
        
        data_min = min(filtered_df['peakP1'].min(), filtered_df['peakP2'].min())
        data_max = max(filtered_df['peakP1'].max(), filtered_df['peakP2'].max())
        ax_subplot.set_xlim(data_min, data_max)
        ax_subplot.set_ylim(data_min, data_max)
        
        ax_subplot.set_xlabel('peakP1')
        ax_subplot.set_ylabel('peakP2')
        ax_subplot.legend()
        
        if df_idx == len(ax) - 1:
            ax_subplot.set_xlim(0, 10.5)
            ax_subplot.set_ylim(0, 10.5)

        # Make spines invisible but keep axis lines
        for spine in ax_subplot.spines.values():
            spine.set_visible(False)
        ax_subplot.get_xaxis().set_ticks_position('none')  # Remove x-ticks but keep axis line
        ax_subplot.get_yaxis().set_ticks_position('none')  # Remove y-ticks but keep axis line

def cell_het_stat(ax):
    all_data = pd.concat(dfs_allcells, ignore_index=True)
    filtered_df = all_data[(all_data['clampPotential'] == -70)]
    formula = 'stpr ~ C(cellID) + C(numPattern)'
    model = ols(formula, data=filtered_df).fit()
    
    anova_table = anova_lm(model, typ=2)
    
    f_stats = anova_table['F']
    p_values = anova_table['PR(>F)']
    
    results_df = pd.DataFrame({'F-statistic': f_stats, 'p-value': p_values})
    
    sns.boxplot(y=results_df['F-statistic'], ax=ax[0])
    ax[0].set_ylabel('F-statistic')
    
    sns.boxplot(y=results_df['p-value'], ax=ax[1])
    ax[1].set_ylabel('p-value')
    
    # Make spines invisible but keep axis lines
    for subplot in ax:
        for spine in subplot.spines.values():
            spine.set_visible(False)
        subplot.get_xaxis().set_ticks_position('none')  # Remove x-ticks but keep axis line
        subplot.get_yaxis().set_ticks_position('none')  # Remove y-ticks but keep axis line

def main():

    fig = plt.figure(figsize=(16, 24))
    
    # Define GridSpec with 4 rows and 6 columns
    gs = gridspec.GridSpec(4, 6, height_ratios=[1, 0.5, 0.75, 1])
    
    # Define the subplots based on the GridSpec layout
    ax = []
    
    # First row: 6 subplots
    for i in range(6):
        ax.append(fig.add_subplot(gs[0, i]))
        
    # Second row: 2 subplots (spanning columns 1 to 2 and 3 to 5)
    ax.append(fig.add_subplot(gs[1, :4]))  # spans columns 0 to 1
    ax.append(fig.add_subplot(gs[1, 4:6])) # spans columns 2 to 3
    
    # Third row: 3 subplots (spanning columns 1 to 3, 4 to 5, and 6)
    ax.append(fig.add_subplot(gs[2, :4]))  # spans columns 0 to 1
    ax.append(fig.add_subplot(gs[2, 4:5])) # spans columns 2 to 3
    ax.append(fig.add_subplot(gs[2, 5:6]))  # spans columns 4 to 5
    
    # Fourth row: 4 subplots
    ax.append(fig.add_subplot(gs[3, :2]))  # spans columns 0 to 1
    ax.append(fig.add_subplot(gs[3, 2:4])) # spans columns 2 to 3
    ax.append(fig.add_subplot(gs[3, 4:5]))  # spans columns 4 to 5
    ax.append(fig.add_subplot(gs[3, 5:6]))  # spans columns 4 to 5
    
    # Plotting functions
    violinplot_cellparams(ax[0:6])
    time_series(ax[6])
    plot_scatter_with_regression_and_r2(ax[7])
    plot_trial_avg_stpr(ax[8:11])
    exemplar_cells(ax[11:13])
    cell_het_stat(ax[13:15])

    # Add panel labels
    labels = 'ABCDEF' + 'GH' + 'IJK' + 'LMNO'
    for i, subplot in enumerate(ax):
        subplot.text(-0.1, 1.1, labels[i], transform=subplot.transAxes, 
                     fontsize=22, fontweight='bold', va='top', ha='right')

    # Remove any unused axes
    for i in range(15, len(ax)):
        fig.delaxes(ax[i])
    
    plt.subplots_adjust(wspace=0.3, hspace=0.6) 
    plt.show()

if __name__ == "__main__":
    main()
