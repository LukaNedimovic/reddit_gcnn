#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns
import pandas as pd
import os 
import ast

from argparser import parse_args


def extract_data_from_csv(csv_path: str=None,
                          data_id:  str=None):
    
    assert csv_path is not None, "CSV path must be provided."
    assert data_id  is not None, "Data ID must be provided."

    df = pd.read_csv(csv_path)

    # From each row, extract model name and relevant data to be plotted
    return df[["model_name", data_id]]


def plot(x_data:   list=None, 
         x_label:  str=None,
         y_data:   list=None,
         y_label:  str=None, 
         title:    str=None,
         png_path: str=None):
    
    sns.set_theme(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_data, y_data, marker="o", linestyle="-", color="#1f77b4", linewidth=2, markersize=8)

    # Adding labels and title with bold text
    plt.xlabel(x_label, fontsize=14, weight="bold")
    plt.ylabel(y_label, fontsize=14, weight="bold")
    plt.title(title,    fontsize=16, weight="bold")

    # Customize ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Optional: Add grid
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Remove the top and right spines for a cleaner look
    sns.despine()

    # Save the plot as a high-resolution PNG for LaTeX inclusion
    plt.savefig(png_path, dpi=300, bbox_inches="tight")


def plot_more(entries,
              x_label:  str=None, 
              y_label:  str=None,
              png_path: str=None):
    
    sns.set_theme(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for idx, entry in entries.iterrows():
        model_name = os.path.basename(entry["model_name"])
        data       = ast.literal_eval(entry[data_id])        

        x_data, y_data = zip(*data)

        # Convert the result to lists
        x_data = list(x_data)
        y_data = list(y_data)

        plt.plot(x_data, y_data, label=model_name)


    # Adding labels and title with bold text
    plt.xlabel(x_label, fontsize=14, weight="bold")
    plt.ylabel(y_label, fontsize=14, weight="bold")

    plt.legend()

    # Customize ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Optional: Add grid
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Remove the top and right spines for a cleaner look
    sns.despine()

    # Save the plot as a high-resolution PNG for LaTeX inclusion
    plt.savefig(png_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # Parse arguments from command line
    args = parse_args("plot")

    csv_path = args.csv_path
    prefix   = args.prefix
    data_id  = args.data_id
    
    x_label  = args.x_label
    y_label  = args.y_label

    # Extract X and Y axis data
    entries = extract_data_from_csv(csv_path, data_id)
    
    # Extract specific groups of entries
    gcn_s_entries  = entries[entries["model_name"].str.startswith("gcn_s_")]
    gcn_m_entries  = entries[entries["model_name"].str.startswith("gcn_m_")]
    gcn_l_entries  = entries[entries["model_name"].str.startswith("gcn_l_")]
    
    sage_s_entries = entries[entries["model_name"].str.startswith("sage_s_")]
    sage_m_entries = entries[entries["model_name"].str.startswith("sage_m_")]
    sage_l_entries = entries[entries["model_name"].str.startswith("sage_l_")]

    # Plot the entries
    plot_more(gcn_s_entries,
              x_label=x_label,
              y_label=y_label,
              png_path=os.path.expandvars(f"$DATA_DIR/plots/grouped/{prefix}_gcn_s_plot.png"))
    
    plot_more(gcn_m_entries,
              x_label=x_label,
              y_label=y_label,
              png_path=os.path.expandvars(f"$DATA_DIR/plots/grouped/{prefix}_gcn_m_plot.png"))
    
    plot_more(gcn_l_entries,
              x_label=x_label,
              y_label=y_label,
              png_path=os.path.expandvars(f"$DATA_DIR/plots/grouped/{prefix}_gcn_l_plot.png"))

    plot_more(sage_s_entries,
              x_label=x_label,
              y_label=y_label,
              png_path=os.path.expandvars(f"$DATA_DIR/plots/grouped/{prefix}_sage_s_plot.png"))

    plot_more(sage_m_entries,
              x_label=x_label,
              y_label=y_label,
              png_path=os.path.expandvars(f"$DATA_DIR/plots/grouped/{prefix}_sage_m_plot.png"))

    plot_more(sage_l_entries,
              x_label=x_label,
              y_label=y_label,
              png_path=os.path.expandvars(f"$DATA_DIR/plots/grouped/{prefix}_sage_l_plot.png"))