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
from scipy.stats import linregress
import matplotlib.gridspec as gridspec

# Set plot parameters
params = {'legend.fontsize': 'x-large', 'figure.figsize': (15, 20), 'axes.labelsize': 'x-large', 
          'axes.titlesize': 'x-large', 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)

hdf5filepath = 'Data_hdf5format_ver3/VC'
hdf5file_list = [file for file in os.listdir(hdf5filepath) if file.endswith("hdf5")]

dfs_allcells = [pd.read_hdf(f"{hdf5filepath}/{file}") for file in hdf5file_list]


def example_traces(ax):
    df = dfs_allcells[10]
    
    df_filtered = df[df['numSQ'] == 15]
    
    # Plot Ch0 data
    df_ch0 = df_filtered.loc[:, 'Ch0_4000':'Ch0_7000']
    df_ch0.T.plot(ax=ax[0], legend=False, color='blue')
    #ax[0].legend([f"Cell ID {df.loc[0, 'cellID']}"], loc='upper right')
    #ax[0].set_title(f"CellID: {df.loc[0, 'cellID']} - Ch0")
    # ax[0].set_xlabel('Time')
    # ax[0].set_ylabel('Value')
    for spine in ax[0].spines.values():
        spine.set_visible(False)
    ax[0].get_xaxis().set_ticks_position('none')
    ax[0].get_yaxis().set_ticks_position('none')
    
    # Plot Ch3 data
    df_ch3 = df_filtered.loc[:, 'Ch3_4000':'Ch3_7000']
    df_ch3.T.plot(ax=ax[1], legend=False, color='red')
    #ax[1].legend([f"Cell ID {df.loc[0, 'cellID']}"], loc='upper right')
    #ax[1].set_title(f"CellID: {df.loc[0, 'cellID']} - Ch3")
    # ax[1].set_xlabel('Time')
    # ax[1].set_ylabel('Value')
    ax[1].set_ylim(-2, -0.5)
    for spine in ax[1].spines.values():
        spine.set_visible(False)
    ax[1].get_xaxis().set_ticks_position('none')
    ax[1].get_yaxis().set_ticks_position('none')

def plot_stpr(ax):
    legend_handles = []
    legend_labels = []
    
    for idx, numSQ in enumerate([1, 5, 15]):
        ax[idx].set_title(f'{numSQ} sq patterns', fontsize=20)
        ax[idx].set_xlabel('Pattern ID', fontsize=20)
        #ax[idx].set_ylim(0, 5)
        ax[idx].tick_params(axis='both', which='major', labelsize=20)
    
        for i, df in enumerate(dfs_allcells):
            df=dfs_allcells[10]
            filtered_df = df[(df['clampPotential'] == -70) & (df['numSQ'] == numSQ)]
            grouped_data = filtered_df.groupby('numPattern')['stpr'].apply(list).reset_index()
    
            x = grouped_data['numPattern']
            y = grouped_data['stpr']
            color_for_df = plt.cm.tab10(i / len(dfs_allcells))
    
            # Plot each peakP1 value
            for pattern_id, stpr_values in zip(x, y):
                ax[idx].scatter([pattern_id] * len(stpr_values), stpr_values, color=color_for_df, label=f'Cell {i+1}' if idx == 0 else "", alpha=0.7, s=150)
    
            if idx == 0:
                legend_handles.append(ax[idx].scatter([], [], color=color_for_df, label=f'Cell {i+1}'))
                legend_labels.append(f'Cell {i+1}')
                
        # Add a horizontal line at y=1 for reference
        ax[idx].axhline(y=1, color='gray', linestyle='--', linewidth=2, label='STPR=1.0')
    
        # Adjust x-axis limits and ticks based on numSQ
        if numSQ == 1:
            ax[idx].set_xlim(0.5, 15.5)
            ax[idx].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        elif numSQ == 5:
            ax[idx].set_xlim(0.5, 3.5)
            ax[idx].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        elif numSQ == 15:
            ax[idx].set_xlim(0.5, 1.5)
            ax[idx].set_xticks([1])
    
    # Add the legend to the first subplot
    # legend = ax[0].legend(legend_handles, legend_labels, title='Colors represent cells', bbox_to_anchor=(0.5, -0.08), loc='upper center', fontsize=15, ncol=len(dfs_allcells), frameon=False, handlelength=0.7, handletextpad=0.3)
    # legend.get_title().set_fontsize('15')
    
    for subplot in ax:
        for spine in subplot.spines.values():
            spine.set_visible(False)
        subplot.get_xaxis().set_ticks_position('none')  # Remove x-ticks but keep axis line
        subplot.get_yaxis().set_ticks_position('none')  # Remove y-ticks but keep axis line

def glm_field(ax):
    combined_df = pd.concat(dfs_allcells, ignore_index=True)
    field_cell_ids = ['2305301', '2306091', '2307171', '2307143', '2308111', '2308171', '2310051']
    field_df = combined_df[combined_df['cellID'].isin(field_cell_ids)]

    model = sm.GLM.from_formula(
        'stpr ~ desensitization',
        data=field_df, family=sm.families.Gaussian())
    results = model.fit()
    #print(results.summary())

    # Extract coefficients and confidence intervals
    params = results.params
    conf = results.conf_int()
    conf['coef'] = params
    conf.columns = ['2.5%', '97.5%', 'Coefficient']
    conf = conf.reset_index().rename(columns={'index': 'Variable'})
    
    # Extract p-values
    p_values = results.pvalues
    conf['p-value'] = p_values
    
    # Define p-value annotation function
    def p_value_annotation(p_value):
        if pd.isna(p_value):
            return 'NaN'  # Indicate missing p-value
        return f'p={p_value:.2e}'  # Format p-value in scientific notation

    conf['P-Value Annotation'] = conf['p-value'].apply(p_value_annotation)
    
    # Plot coefficients and confidence intervals
    sns.barplot(x='Coefficient', y='Variable', data=conf, color='skyblue', ax=ax[0], errorbar=None)
    ax[0].errorbar(conf['Coefficient'], conf['Variable'],
                   xerr=[conf['Coefficient'] - conf['2.5%'], conf['97.5%'] - conf['Coefficient']],
                   fmt='o', color='black', capsize=5)
    
    # Annotate p-values on the plot
    for i in range(len(conf)):
        p_value_text = conf['P-Value Annotation'].iloc[i]
        ax[0].text(conf['Coefficient'].iloc[i], conf['Variable'].iloc[i], 
                   p_value_text, va='center', ha='left', fontsize=10, color='black')

    ax[0].axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax[0].set_xlabel('Coefficient')
    ax[0].set_title('GLM Coefficients with 95% Confidence Intervals and P-Values')

    # Prepare data for pie chart
    contribution_df = pd.DataFrame({
        'Variable': conf['Variable'],
        'Coefficient': conf['Coefficient'],
        'p-value': conf['p-value']
    })
    
    # Calculate the percent variance explained by all predictors
    total_variance_explained = contribution_df['Coefficient'].abs().sum()
    total_variance = total_variance_explained  # Total variance is the sum of all coefficients' absolute values

    if total_variance == 0:
        total_variance = 1  # Avoid division by zero

    contributions_percent = {row['Variable']: abs(row['Coefficient']) / total_variance * 100 for _, row in contribution_df.iterrows()}

    # Calculate other contributors' value
    other_contributors_value = 100 - sum(contributions_percent.values())
    
    # Ensure no negative values
    if other_contributors_value < 0:
        other_contributors_value = 0
    
    pie_data = {**contributions_percent, 'Other Predictors': other_contributors_value}

    # Plot pie chart
    ax[1].pie(pie_data.values(), labels=pie_data.keys(), autopct='%1.1f%%', colors=plt.cm.Paired(range(len(pie_data))), startangle=140)
    ax[1].axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    ax[1].set_title('Percent Contributions of Predictors to Variance Explained')


def field_vs_stpr(ax):
    
    colors = {1: '#E69F00',  # Orange
            5: '#56B4E9',  # Blue
            15: '#009E73'}  # Teal
    numSQ_labels = {1: '1SQ', 5: '5SQ', 15: '15SQ'}
    
    for i, (clamp_potential, title) in enumerate([(-70, 'Excitation'), (0, 'Inhibition')]):
        ax_ = ax[i]
        
        combined_df = pd.concat(dfs_allcells, ignore_index=True)
        field_cell_ids = ['2305301', '2306091', '2307171', '2307143', '2308111', '2308171', '2310051']
        field_df = combined_df[combined_df['cellID'].isin(field_cell_ids)] 
        filtered_df = field_df[field_df['clampPotential'] == clamp_potential]
        
        for numSQ in [1, 5, 15]:
            df_numSQ = filtered_df[filtered_df['numSQ'] == numSQ]
            x = df_numSQ['desensitization']
            y = df_numSQ['stpr']
            
            
            ax_.scatter(x, y, color=colors[numSQ], alpha=0.9, label=numSQ_labels[numSQ])
        
        ax_.set_title(title, fontsize=20)
        ax_.set_xlabel('desensitization', fontsize=16)
        ax_.set_ylabel('stpr', fontsize=16)
        
        # Add a legend
        ax_.legend(title='numSQ', fontsize=14)
        
        # Make spines invisible but keep axis lines
        for spine in ax_.spines.values():
            spine.set_visible(False)
        ax_.get_xaxis().set_ticks_position('none')
        ax_.get_yaxis().set_ticks_position('none')


def cell_physio_vs_stpr(ax):
    conditions = {'clampPotential': [-70, 0]}  # Excitation: -70, Inhibition: 0
    
    color_map = plt.get_cmap('tab10')
    
    for i, clampPotential in enumerate(conditions['clampPotential']):
        filtered_dfs = [df[(df['numSQ'] == 15) & (df['clampPotential'] == clampPotential)] for df in dfs_allcells]
        filtered_df = pd.concat(filtered_dfs)
        
        # P1 vs HoldI
        ax[2*i].set_title(f'P1 vs HoldI\n{"Excitation" if clampPotential == -70 else "Inhibition"}', fontsize=16)
        ax[2*i].set_xlabel('Holding Current (pA)', fontsize=14)
        ax[2*i].set_ylabel('stpr', fontsize=14)
        
        legend_lines = []
        legend_labels = []
        
        for k, df in enumerate(dfs_allcells):
            df_filtered = df[(df['numSQ'] == 15) & (df['clampPotential'] == clampPotential)]
            if not df_filtered.empty:
                color = color_map(k % 10)
                
                # Scatter plot with points
                ax[2*i].scatter(df_filtered['holdI'], df_filtered['stpr'], color=color, alpha=0.7)
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = linregress(df_filtered['holdI'], df_filtered['stpr'])
                line, = ax[2*i].plot(df_filtered['holdI'], slope * df_filtered['holdI'] + intercept, color=color)
                
                # Add R² value to legend
                legend_lines.append(line)
                legend_labels.append(f'R²={r_value**2:.2f}')
        
        ax[2*i].legend(legend_lines, legend_labels, loc='upper left', fontsize=10)
        
        # Remove spines for the first two plots
        if i == 0:
            for spine in ax[2*i].spines.values():
                spine.set_visible(False)
            ax[2*i].get_xaxis().set_ticks_position('none')
            ax[2*i].get_yaxis().set_ticks_position('none')
        
        # IR vs PeakP1
        ax[2*i+1].set_title(f'IR vs PeakP1\n{"Excitation" if clampPotential == -70 else "Inhibition"}', fontsize=16)
        ax[2*i+1].set_xlabel('IR', fontsize=14)
        ax[2*i+1].set_ylabel('stpr', fontsize=14)
        
        legend_lines = []
        legend_labels = []
        
        for k, df in enumerate(dfs_allcells):
            df_filtered = df[(df['numSQ'] == 15) & (df['clampPotential'] == clampPotential)]
            if not df_filtered.empty:
                color = color_map(k % 10)
                
                # Scatter plot with points
                ax[2*i+1].scatter(df_filtered['IR'], df_filtered['stpr'], color=color, alpha=0.7)
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = linregress(df_filtered['IR'], df_filtered['stpr'])
                line, = ax[2*i+1].plot(df_filtered['IR'], slope * df_filtered['IR'] + intercept, color=color)
                
                # Add R² value to legend
                legend_lines.append(line)
                legend_labels.append(f'R²={r_value**2:.2f}')
        
        ax[2*i+1].legend(legend_lines, legend_labels, loc='upper left', fontsize=10)
        
        # Remove spines for the first two plots
    for ax_ in ax:
        for spine in ax_.spines.values():
            spine.set_visible(False)
        ax_.get_xaxis().set_ticks_position('none')
        ax_.get_yaxis().set_ticks_position('none')

def glm_cell_physio(ax):
    combined_df_EI = pd.concat(dfs_allcells, ignore_index=True)

    filtered_dfs = []
    for df in dfs_allcells:
        # Create a unique ID by combining 'cellID' and 'numSQ'
        df['unique_ID'] = (
            df['cellID'].astype(str) + '_' + df['EorI'].astype(str) + '_' +
            df['numSQ'].astype(str) + '_' +
            df['numPattern'].astype(str)
        )
        filtered_dfs.append(df)
    combined_df = pd.concat(filtered_dfs, ignore_index=True)
    
    # Define and fit the GLM model
    model = sm.GLM.from_formula(
        'stpr ~ C(numSQ) + holdI + IR + desensitization + bln_std_cell + snr_cell + peakP1',
        data=combined_df, family=sm.families.Gaussian())
    results = model.fit()
    print(results.summary())
    
    # Extract coefficients and confidence intervals
    params = results.params
    conf = results.conf_int()
    conf['coef'] = params
    conf.columns = ['2.5%', '97.5%', 'Coefficient']
    conf = conf.reset_index().rename(columns={'index': 'Variable'})
    
    # Extract p-values
    p_values = results.pvalues
    conf['p-value'] = p_values
    
    # Define p-value annotation function
    def p_value_annotation(p_value):
        if pd.isna(p_value):
            return 'NaN'  # Indicate missing p-value
        return f'p={p_value:.2e}'  # Format p-value in scientific notation

    conf['P-Value Annotation'] = conf['p-value'].apply(p_value_annotation)
    
    # Plot coefficients and confidence intervals
    sns.barplot(x='Coefficient', y='Variable', data=conf, color='skyblue', ax=ax[0], errorbar=None)
    ax[0].errorbar(conf['Coefficient'], conf['Variable'],
                   xerr=[conf['Coefficient'] - conf['2.5%'], conf['97.5%'] - conf['Coefficient']],
                   fmt='o', color='black', capsize=5)
    
    # Annotate p-values on the plot
    for i in range(len(conf)):
        p_value_text = conf['P-Value Annotation'].iloc[i]
        ax[0].text(conf['Coefficient'].iloc[i], conf['Variable'].iloc[i], 
                   p_value_text, va='center', ha='left', fontsize=10, color='black')
    
    ax[0].axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax[0].set_xlabel('Coefficient')
    ax[0].set_title('GLM Coefficients with 95% Confidence Intervals and P-Values')
    
    # Prepare data for pie chart
    contribution_df = pd.DataFrame({
        'Variable': conf['Variable'],
        'Coefficient': conf['Coefficient'],
        'p-value': conf['p-value']
    })
    
    # Calculate the percent variance explained by all predictors
    total_variance_explained = contribution_df['Coefficient'].abs().sum()
    total_variance = total_variance_explained  # Total variance is the sum of all coefficients' absolute values

    if total_variance == 0:
        total_variance = 1  # Avoid division by zero

    contributions_percent = {row['Variable']: abs(row['Coefficient']) / total_variance * 100 for _, row in contribution_df.iterrows()}

    # Calculate other contributors' value
    other_contributors_value = 100 - sum(contributions_percent.values())
    
    # Ensure no negative values
    if other_contributors_value < 0:
        other_contributors_value = 0
    
    pie_data = {**contributions_percent, 'Other Predictors': other_contributors_value}

    # # Plot pie chart
    # ax[2].pie(pie_data.values(), labels=pie_data.keys(), autopct='%1.1f%%', colors=plt.cm.Paired(range(len(pie_data))), startangle=140)
    # ax[2].axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    # ax[2].set_title('Percent Contributions of Predictors to Variance Explained')
    
    # Create a new figure for the pie chart
    ax[1].pie(pie_data.values(), labels=pie_data.keys(), autopct='%1.1f%%', colors=plt.cm.Paired(range(len(pie_data))), startangle=140)
    ax[1].axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    ax[1].set_title('Percent Contributions of Predictors to Variance Explained')


def main():

    fig = plt.figure(figsize=(16, 24))
    
    # Define GridSpec with 4 rows and 6 columns
    gs = gridspec.GridSpec(4, 8, height_ratios=[1, 0.5, 0.75, 1])
    
    # Define the subplots based on the GridSpec layout
    ax = []
    
    # First row: 5 subplots
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[0, 1])) 
    ax.append(fig.add_subplot(gs[0, 2:6]))
    ax.append(fig.add_subplot(gs[0, 6]))
    ax.append(fig.add_subplot(gs[0, 7]))
    
    # Second row: 2 subplots (spanning columns 0 to 4 and 4 to 6)
    ax.append(fig.add_subplot(gs[1, :2])) 
    ax.append(fig.add_subplot(gs[1, 2:4])) 
    ax.append(fig.add_subplot(gs[1, 5:7]))
    ax.append(fig.add_subplot(gs[1, 7]))
    
    # Third row: 3 subplots (spanning columns 0 to 4, 4 to 5, and 5 to 6)
    ax.append(fig.add_subplot(gs[2, :2])) 
    ax.append(fig.add_subplot(gs[2, 2:4]))  
    ax.append(fig.add_subplot(gs[2, 4:6]))  
    ax.append(fig.add_subplot(gs[2, 6:8]))
    
    # Fourth row: 4 subplots
    ax.append(fig.add_subplot(gs[3, :3]))
    ax.append(fig.add_subplot(gs[3, 3:5]))
    
    # Plotting functions
    example_traces(ax[0:2])
    plot_stpr(ax[2:5])
    field_vs_stpr(ax[5:7])
    glm_field(ax[7:9])
    cell_physio_vs_stpr(ax[9:13])
    glm_cell_physio(ax[13:16])
    
    # Add panel labels
    labels = 'ABCDEFGHIJKLMNOPQ'
    for i, subplot in enumerate(ax):
        subplot.text(-0.1, 1.1, labels[i], transform=subplot.transAxes, 
                     fontsize=22, fontweight='bold', va='top', ha='right')

    # Remove any unused axes
    for i in range(len(ax), len(fig.axes)):
        fig.delaxes(fig.axes[i])
        
    plt.subplots_adjust(wspace=0.7, hspace=0.4, top=0.95, bottom=0.05, left=0.07, right=0.93)
    plt.show()


if __name__ == "__main__":
    main()
