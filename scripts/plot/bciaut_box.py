import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
from bciaut import load_subject_data

def generate_plot():
    df_ours = load_subject_data("MSR_P300NET (Ours)", "msr_p300net")
    if df_ours is None:
        df_ours = load_subject_data("MSR_P300NET (Ours)", "normal_hst")
    df_eegnet = load_subject_data("EEGNetv4", "eegnet")
    df_p3cnet = load_subject_data("P3CNET", "p3cnet")
    
    if df_ours is None or df_eegnet is None or df_p3cnet is None:
        print("Warning: Missing BCIAUT subject results. Skipping Box Plot.")
        return
        
    df_all = pd.concat([df_ours, df_eegnet, df_p3cnet], ignore_index=True)
    
    df_melt = df_all.melt(id_vars=['Subject', 'Model'], value_vars=['Obj Acc (%)', 'Macro F1'],
                          var_name='Metric', value_name='Value')
    df_melt.loc[df_melt['Metric'] == 'Macro F1', 'Value'] *= 100.0
    df_melt['Metric'] = df_melt['Metric'].map({'Obj Acc (%)': 'Accuracy (%)', 'Macro F1': 'Macro F1 (x100)'})
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    model_order = ["MSR_P300NET (Ours)", "EEGNetv4", "P3CNET"]
    sns.boxplot(data=df_melt, x='Metric', y='Value', hue='Model', hue_order=model_order, ax=ax,
                palette=config.MODEL_COLORS, width=0.55, gap=0.15, showfliers=True,
                boxprops=dict(alpha=0.85, linewidth=2.0),
                medianprops=dict(linewidth=2.5, color='black'),
                whiskerprops=dict(linewidth=2.0),
                capprops=dict(linewidth=2.0))
                
    config.apply_theme(ax)
    ax.set_ylabel("Performance Score", fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    ax.set_xlabel("")
    ax.set_ylim(40, 105)
    
    # Legend formatting
    legend = ax.legend(frameon=True, facecolor='white', edgecolor='none', prop={'size': config.LEGEND_SIZE, 'family': 'Times New Roman'})
    
    plt.tight_layout()
    save_pdf = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_box_metrics.pdf")
    save_png = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_box_metrics.png")
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")

if __name__ == "__main__":
    generate_plot()
