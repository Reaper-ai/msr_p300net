import os
import pandas as pd
import matplotlib.pyplot as plt

import config
from gist import load_itr_data

def generate_plot():
    df_ours = load_itr_data("MSR_P300NET (Ours)", "msr_p300net")
    df_eegnet = load_itr_data("EEGNetv4", "eegnet")
    df_eeginc = load_itr_data("EEGInception", "eeginception")
    
    if df_ours is None or df_eegnet is None or df_eeginc is None:
        print("Warning: Missing GIST ITR results. Skipping ITR lines.")
        return
        
    df_all = pd.concat([df_ours, df_eegnet, df_eeginc], ignore_index=True)
    
    # Plot 1: Accuracy vs Sequences
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model_name in df_all['Model'].unique():
        df_sub = df_all[df_all['Model'] == model_name]
        ax.plot(df_sub['Sequence'], df_sub['Accuracy (%)'], marker='o', label=model_name,
                color=config.MODEL_COLORS[model_name], linewidth=2.5, markersize=8)
    config.apply_theme(ax)
    ax.set_xlabel("Sequence Repetitions", fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    ax.set_ylabel("Spelling Accuracy (%)", fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    ax.set_xticks(range(1, 16))
    ax.legend(frameon=True, facecolor='white', edgecolor='none', prop={'size': config.LEGEND_SIZE, 'family': 'Times New Roman'})
    plt.tight_layout()
    save_pdf = os.path.join(config.PLOT_DIR_GIST, "gist_itr_accuracy.pdf")
    save_png = os.path.join(config.PLOT_DIR_GIST, "gist_itr_accuracy.png")
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")
    
    # Plot 2: ITR vs Sequences
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model_name in df_all['Model'].unique():
        df_sub = df_all[df_all['Model'] == model_name]
        ax.plot(df_sub['Sequence'], df_sub['ITR (bits/select)'], marker='o', label=model_name,
                color=config.MODEL_COLORS[model_name], linewidth=2.5, markersize=8)
    config.apply_theme(ax)
    ax.set_xlabel("Sequence Repetitions", fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    ax.set_ylabel("ITR (bits/selection)", fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    ax.set_xticks(range(1, 16))
    ax.legend(frameon=True, facecolor='white', edgecolor='none', prop={'size': config.LEGEND_SIZE, 'family': 'Times New Roman'})
    plt.tight_layout()
    save_pdf = os.path.join(config.PLOT_DIR_GIST, "gist_itr_bitrate.pdf")
    save_png = os.path.join(config.PLOT_DIR_GIST, "gist_itr_bitrate.png")
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")

if __name__ == "__main__":
    generate_plot()
