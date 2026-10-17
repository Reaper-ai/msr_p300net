import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

def get_subcohort(row):
    if row['Condition'] == 'Healthy':
        return 'Healthy'
    elif row['Dataset_ID'] == 2 and row['Condition'] == 'Unhealthy':
        return 'MS Only'
    else:
        return 'Other Unh.'

def generate_plot():
    csv_path = os.path.join(config.ROOT_DIR, "results", "gib", "msr_p300net", "subject_results.csv")
    if not os.path.exists(csv_path):
        print("Warning: Missing GIB msr_p300net subject results. Skipping Box Plot.")
        return
        
    df = pd.read_csv(csv_path)
    # Average across seeds per subject first
    df_subj = df.groupby(['Subject_ID', 'Dataset_ID', 'Condition'])[['Obj Acc (%)', 'Macro F1']].mean().reset_index()
    
    # Assign cohorts
    df_subj['Cohort'] = df_subj.apply(get_subcohort, axis=1)
    
    # Duplicate for 'All' cohort
    df_all = df_subj.copy()
    df_all['Cohort'] = 'All'
    
    df_plot = pd.concat([df_all, df_subj], ignore_index=True)
    
    # Melt metrics into uniform percentage scale
    df_melt = df_plot.melt(id_vars=['Subject_ID', 'Cohort'], value_vars=['Obj Acc (%)', 'Macro F1'],
                           var_name='Metric', value_name='Score')
    df_melt.loc[df_melt['Metric'] == 'Macro F1', 'Score'] *= 100.0
    df_melt['Metric'] = df_melt['Metric'].map({'Obj Acc (%)': 'Accuracy', 'Macro F1': 'Macro F1'})
    
    cohort_order = ['All', 'Healthy', 'MS Only', 'Other Unh.']
    
    # IEEE Single-Column Compact Dimensions
    fig, ax = plt.subplots(figsize=(config.IEEE_SINGLE_COL_WIDTH, 2.5))
    
    metric_colors = {
        'Accuracy': config.COLOR_OUR_MODEL,
        'Macro F1': config.COLOR_F1
    }
    
    sns.boxplot(data=df_melt, x='Cohort', y='Score', hue='Metric', order=cohort_order, ax=ax,
                palette=metric_colors, width=0.55, gap=0.15, showfliers=True,
                boxprops=dict(alpha=0.8, linewidth=0.9),
                medianprops=dict(linewidth=1.2, color='#111111'),
                whiskerprops=dict(linewidth=0.9),
                capprops=dict(linewidth=0.9),
                flierprops=dict(marker='o', markersize=2.5, alpha=0.5))
                
    config.apply_ieee_theme(ax)
    
    ax.set_xlabel("")
    ax.set_ylabel("Score (%)", fontsize=config.IEEE_LABEL_SIZE, fontname='Arial', labelpad=2)
    ax.set_ylim(35, 105)
    
    # Minimalist legend in bottom left
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#e0e0e0',
              fontsize=config.IEEE_LEGEND_SIZE-1.0, prop={'size': config.IEEE_LEGEND_SIZE-1.0, 'family': 'Arial'})
    
    plt.tight_layout()
    save_pdf = os.path.join(config.PLOT_DIR_GIB, "gib_box_cohort_metrics.pdf")
    save_png = os.path.join(config.PLOT_DIR_GIB, "gib_box_cohort_metrics.png")
    plt.savefig(save_pdf, dpi=350, bbox_inches='tight')
    plt.savefig(save_png, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")

if __name__ == "__main__":
    generate_plot()
