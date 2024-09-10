"""
Author: Sulu
Python version: 3.9
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import matplotlib.pylab as pylab


# Set plot parameters
params = {
    'legend.fontsize': 'x-large',
    'figure.figsize': (15, 20),
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large'
}
pylab.rcParams.update(params)

def ridgeline_density_plot(dfs_allcells):
    combined_df = pd.concat(dfs_allcells, ignore_index=True)
    exc_df = combined_df[combined_df['clampPotential'] == -70]
    inh_df = combined_df[combined_df['clampPotential'] == 0]
    field_cell_ids = ['2305301', '2306091', '2307171', '2307143', '2308111', '2308171', '2310051']
    field_df = combined_df[combined_df['cellID'].isin(field_cell_ids)]
    print("Creating plots...")
    fig, axs = plt.subplots(nrows=6, sharex=True, sharey=False, figsize=(15, 20))

    bw = 0.1
    colors = {
        15: 'blue',
        5: 'orange',
        1: 'green'
    }
    labels = {
        15: '15 SQ',
        5: '5 SQ',
        1: '1 SQ'
    }

    kde_params = {
        'bw_adjust': bw,
        'fill': True,
        'alpha': 0.4,
        'linewidth': 2.5  # Set linewidth to make lines darker and thicker
    }

    for numSQ in [15, 5, 1]:
        subset_df = inh_df[inh_df['numSQ'] == numSQ]
        if not subset_df.empty:
            color = colors[numSQ]
            sns.kdeplot(
                data=subset_df,
                x='peakP1',
                color=color,
                ax=axs[0],
                label=labels[numSQ],
                **kde_params
            )

    for numSQ in [15, 5, 1]:
        subset_df = inh_df[inh_df['numSQ'] == numSQ]
        if not subset_df.empty:
            color = colors[numSQ]
            sns.kdeplot(
                data=subset_df,
                x='peakP2',
                color=color,
                ax=axs[1],
                label=labels[numSQ],
                **kde_params
            )

    for numSQ in [15, 5, 1]:
        subset_df = exc_df[exc_df['numSQ'] == numSQ]
        if not subset_df.empty:
            color = colors[numSQ]
            sns.kdeplot(
                data=subset_df,
                x='peakP1',
                color=color,
                ax=axs[2],
                label=labels[numSQ],
                **kde_params
            )

    for numSQ in [15, 5, 1]:
        subset_df = exc_df[exc_df['numSQ'] == numSQ]
        if not subset_df.empty:
            color = colors[numSQ]
            sns.kdeplot(
                data=subset_df,
                x='peakP2',
                color=color,
                ax=axs[3],
                label=labels[numSQ],
                **kde_params
            )

    for numSQ in [15, 5, 1]:
        subset_df = field_df[field_df['numSQ'] == numSQ]
        if not subset_df.empty:
            color = colors[numSQ]
            sns.kdeplot(
                data=subset_df,
                x='field_ptp_P1',
                color=color,
                ax=axs[4],
                label=labels[numSQ],
                **kde_params
            )

    for numSQ in [15, 5, 1]:
        subset_df = field_df[field_df['numSQ'] == numSQ]
        if not subset_df.empty:
            color = colors[numSQ]
            sns.kdeplot(
                data=subset_df,
                x='field_ptp_P2',
                color=color,
                ax=axs[5],
                label=labels[numSQ],
                **kde_params
            )

    axs[0].set_ylabel('Peak1 Inh')
    axs[1].set_ylabel('Peak2 Inh')
    axs[2].set_ylabel('Peak1 Exc')
    axs[3].set_ylabel('Peak2 Exc')
    axs[4].set_ylabel('Field Peak1')
    axs[5].set_ylabel('Field Peak2')
    axs[5].set_xlabel('Current (pA)')

    for ax in axs:
        ax.set_xlim(-10, 150)

    axs[0].legend(title='num SQ')

    plt.tight_layout()
    plt.show()

def ridgeline_histogram_plot(dfs_allcells):
    colors = {
        15: 'blue',
        5: 'orange',
        1: 'green'
    }
    labels = {
        15: '15 SQ',
        5: '5 SQ',
        1: '1 SQ'
    }

    hist_params = {
        'binwidth': 5,
        'alpha': 0.4,
        'linewidth': 2.5
    }

    combined_df = pd.concat(dfs_allcells, ignore_index=True)
    exc_df = combined_df[combined_df['clampPotential'] == -70]
    inh_df = combined_df[combined_df['clampPotential'] == 0]
    field_cell_ids = ['2305301', '2306091', '2307171', '2307143', '2308111', '2308171', '2310051']
    field_df = combined_df[combined_df['cellID'].isin(field_cell_ids)]

    fig, axs = plt.subplots(nrows=6, sharex=True, sharey=True, figsize=(15, 20))

    for numSQ in [15, 5, 1]:
        subset_df = inh_df[inh_df['numSQ'] == numSQ]
        if not subset_df.empty:
            color = colors[numSQ]
            sns.histplot(
                data=subset_df,
                x='peakP1',
                color=color,
                ax=axs[0],
                label=labels[numSQ],
                **hist_params
            )

    for numSQ in [15, 5, 1]:
        subset_df = inh_df[inh_df['numSQ'] == numSQ]
        if not subset_df.empty:
            color = colors[numSQ]
            sns.histplot(
                data=subset_df,
                x='peakP2',
                color=color,
                ax=axs[1],
                label=labels[numSQ],
                **hist_params
            )

    for numSQ in [15, 5, 1]:
        subset_df = exc_df[exc_df['numSQ'] == numSQ]
        if not subset_df.empty:
            color = colors[numSQ]
            sns.histplot(
                data=subset_df,
                x='peakP1',
                color=color,
                ax=axs[2],
                label=labels[numSQ],
                **hist_params
            )

    for numSQ in [15, 5, 1]:
        subset_df = exc_df[exc_df['numSQ'] == numSQ]
        if not subset_df.empty:
            color = colors[numSQ]
            sns.histplot(
                data=subset_df,
                x='peakP2',
                color=color,
                ax=axs[3],
                label=labels[numSQ],
                **hist_params
            )

    for numSQ in [15, 5, 1]:
        subset_df = field_df[field_df['numSQ'] == numSQ]
        if not subset_df.empty:
            color = colors[numSQ]
            sns.histplot(
                data=subset_df,
                x='field_ptp_P1',
                color=color,
                ax=axs[4],
                label=labels[numSQ],
                **hist_params
            )

    for numSQ in [15, 5, 1]:
        subset_df = field_df[field_df['numSQ'] == numSQ]
        if not subset_df.empty:
            color = colors[numSQ]
            sns.histplot(
                data=subset_df,
                x='field_ptp_P2',
                color=color,
                ax=axs[5],
                label=labels[numSQ],
                **hist_params
            )

    axs[0].set_ylabel('Peak1 Inh')
    axs[1].set_ylabel('Peak2 Inh')
    axs[2].set_ylabel('Peak1 Exc')
    axs[3].set_ylabel('Peak2 Exc')
    axs[4].set_ylabel('Field Peak1')
    axs[5].set_ylabel('Field Peak2')
    axs[5].set_xlabel('Current (pA)')

    for ax in axs:
        ax.set_xlim(-10, 150)

    axs[0].legend(title='num SQ')

    plt.tight_layout()
    plt.show()

def main():
    hdf5filepath = input("Enter the directory path for the HDF5 files: ")

    if not os.path.exists(hdf5filepath):
        print("HDF5 file not found")
        return

    hdf5file_list = [file for file in os.listdir(hdf5filepath) if file.endswith("hdf5")]

    if not hdf5file_list:
        print("No HDF5 files found in the specified directory")
        return

    dfs_allcells = []
    for file in hdf5file_list:
        df = pd.read_hdf(f"{hdf5filepath}/{file}")
        dfs_allcells.append(df)

    ridgeline_density_plot(dfs_allcells)
    ridgeline_histogram_plot(dfs_allcells)

if __name__ == "__main__":
    main()
