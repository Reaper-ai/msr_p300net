import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
from gist import load_subject_data

def generate_plot():
    model_order = ["MSR_P300NET (Ours)", "EEGNetv4", "EEGInception"]

    # 1. GIST RSVP Box Plots (Dropping 'Single BalAcc (%)' as requested)
    df_ours_rsvp = load_subject_data("MSR_P300NET (Ours)", "rsvp", "msr_p300net")
    df_eegnet_rsvp = load_subject_data("EEGNetv4", "rsvp", "eegnet")
    df_eeginc_rsvp = load_subject_data("EEGInception", "rsvp", "eeginception")
    
    if df_ours_rsvp is not None and df_eegnet_rsvp is not None and df_eeginc_rsvp is not None:
        df_all_rsvp = pd.concat([df_ours_rsvp, df_eegnet_rsvp, df_eeginc_rsvp], ignore_index=True)
        
        # Melt evaluation metrics excluding 'Single BalAcc (%)'
        df_melt = df_all_rsvp.melt(id_vars=['Subject', 'Model'],
                                   value_vars=['Macro F1', 'AUC', 'Seq Top-1 Acc (%)'],
                                   var_name='Metric', value_name='Value')
        
        # Scale F1 and AUC by 100.0 for uniform plotting range
        df_melt.loc[df_melt['Metric'] == 'Macro F1', 'Value'] *= 100.0
        df_melt.loc[df_melt['Metric'] == 'AUC', 'Value'] *= 100.0
        
        # Friendly labels
        df_melt['Metric'] = df_melt['Metric'].map({
            'Macro F1': 'Macro F1 (x100)',
            'AUC': 'AUC (x100)',
            'Seq Top-1 Acc (%)': 'Seq Top-1 Acc (%)'
        })
        
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.boxplot(data=df_melt, x='Metric', y='Value', hue='Model', hue_order=model_order, ax=ax,
                    palette=config.MODEL_COLORS, width=0.55, gap=0.15, showfliers=True,
                    boxprops=dict(alpha=0.85, linewidth=2.0),
                    medianprops=dict(linewidth=2.5, color='black'),
                    whiskerprops=dict(linewidth=2.0),
                    capprops=dict(linewidth=2.0))
                    
        config.apply_theme(ax)
        ax.set_ylabel("Metric Score", fontsize=config.LABEL_SIZE, fontname='Times New Roman')
        ax.set_xlabel("")
        ax.set_ylim(30, 105)
        ax.legend(frameon=True, facecolor='white', edgecolor='none', prop={'size': config.LEGEND_SIZE, 'family': 'Times New Roman'})
        
        plt.tight_layout()
        save_pdf = os.path.join(config.PLOT_DIR_GIST, "gist_rsvp_box_metrics.pdf")
        save_png = os.path.join(config.PLOT_DIR_GIST, "gist_rsvp_box_metrics.png")
        plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
        plt.savefig(save_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_pdf} and {save_png}")
    else:
        print("Warning: Missing GIST RSVP results. Skipping RSVP Box plot.")

    # 2. GIST P300 Box Plots (Dropping 'Trial BalAcc (%)' as requested)
    df_ours_p300 = load_subject_data("MSR_P300NET (Ours)", "p300", "msr_p300net")
    df_eegnet_p300 = load_subject_data("EEGNetv4", "p300", "eegnet")
    df_eeginc_p300 = load_subject_data("EEGInception", "p300", "eeginception")
    
    if df_ours_p300 is not None and df_eegnet_p300 is not None and df_eeginc_p300 is not None:
        df_all_p300 = pd.concat([df_ours_p300, df_eegnet_p300, df_eeginc_p300], ignore_index=True)
        
        # Melt evaluation metrics excluding 'Trial BalAcc (%)'
        df_melt = df_all_p300.melt(id_vars=['Subject', 'Model'],
                                   value_vars=['Macro F1', 'Trial AUC', 'Pairwise Acc (%)'],
                                   var_name='Metric', value_name='Value')
                                   
        # Scale F1 and AUC by 100.0 for uniform plotting range
        df_melt.loc[df_melt['Metric'] == 'Macro F1', 'Value'] *= 100.0
        df_melt.loc[df_melt['Metric'] == 'Trial AUC', 'Value'] *= 100.0
        
        # Friendly labels
        df_melt['Metric'] = df_melt['Metric'].map({
            'Macro F1': 'Macro F1 (x100)',
            'Trial AUC': 'Trial AUC (x100)',
            'Pairwise Acc (%)': 'Pairwise Acc (%)'
        })
        
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.boxplot(data=df_melt, x='Metric', y='Value', hue='Model', hue_order=model_order, ax=ax,
                    palette=config.MODEL_COLORS, width=0.55, gap=0.15, showfliers=True,
                    boxprops=dict(alpha=0.85, linewidth=2.0),
                    medianprops=dict(linewidth=2.5, color='black'),
                    whiskerprops=dict(linewidth=2.0),
                    capprops=dict(linewidth=2.0))
                    
        config.apply_theme(ax)
        ax.set_ylabel("Metric Score", fontsize=config.LABEL_SIZE, fontname='Times New Roman')
        ax.set_xlabel("")
        ax.set_ylim(50, 105)
        ax.legend(frameon=True, facecolor='white', edgecolor='none', prop={'size': config.LEGEND_SIZE, 'family': 'Times New Roman'})
        
        plt.tight_layout()
        save_pdf = os.path.join(config.PLOT_DIR_GIST, "gist_p300_box_metrics.pdf")
        save_png = os.path.join(config.PLOT_DIR_GIST, "gist_p300_box_metrics.png")
        plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
        plt.savefig(save_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_pdf} and {save_png}")
    else:
        print("Warning: Missing GIST P300 results. Skipping P300 Box plot.")

    # 3. Ours-only combined box plot: single axes, P300 (3 boxes) + gap + RSVP (3 boxes)
    df_ours_rsvp = load_subject_data("MSR_P300NET (Ours)", "rsvp", "msr_p300net")
    df_ours_p300 = load_subject_data("MSR_P300NET (Ours)", "p300", "msr_p300net")

    if df_ours_rsvp is None or df_ours_p300 is None:
        print("Warning: Missing GIST ours results. Skipping ours-only box plot.")
        return

    metric_colors = {'Accuracy': config.COLOR_OUR_MODEL,
                     'Macro F1': config.COLOR_EEGNET,
                     'AUC': config.COLOR_P3CNET}
    task_specs = [
        ('P300', df_ours_p300, [('Pairwise Acc (%)', 'Accuracy'), ('Macro F1', 'Macro F1'), ('Trial AUC', 'AUC')]),
        ('RSVP', df_ours_rsvp, [('Seq Top-1 Acc (%)', 'Accuracy'), ('Macro F1', 'Macro F1'), ('AUC', 'AUC')]),
    ]

    parts = []
    x_pos = 0
    pos_map = {}
    for task, df_task, metrics in task_specs:
        for col, label in metrics:
            s = df_task[['Subject', col]].copy()
            if label in ('Macro F1', 'AUC'):
                s[col] *= 100.0
            s = s.rename(columns={col: 'Value'})
            s['Metric'] = label
            s['Task'] = task
            s['xpos'] = x_pos
            parts.append(s)
            pos_map[(task, label)] = x_pos
            x_pos += 0.7   # tight intra-group spacing
        x_pos += 0.8  # larger gap between P300 and RSVP groups

    df_ours_all = pd.concat(parts, ignore_index=True)

    # Stats per position; draw with ax.bxp at explicit x (avoids seaborn categorical drift)
    stats = []
    colors = []
    for (task, metric), xpos in sorted(pos_map.items(), key=lambda kv: kv[1]):
        v = df_ours_all.loc[df_ours_all['xpos'] == xpos, 'Value']
        q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        inliers = v[(v >= lo) & (v <= hi)]
        stats.append({'label': '', 'med': med, 'q1': q1, 'q3': q3,
                      'whislo': inliers.min(), 'whishi': inliers.max(),
                      'fliers': v[(v < lo) | (v > hi)].values, 'mean': v.mean()})
        colors.append(metric_colors[metric])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    positions = sorted(pos_map.values())
    ax.bxp(stats, positions=positions, widths=0.55, showfliers=True,
           patch_artist=True, manage_ticks=False,
           boxprops=dict(linewidth=1.5, edgecolor='#222222'),
           medianprops=dict(linewidth=1.8, color='black'),
           whiskerprops=dict(linewidth=1.5, color='#222222'),
           capprops=dict(linewidth=1.5, color='#222222'),
           flierprops=dict(marker='o', markersize=4, markerfacecolor='none',
                           markeredgecolor='#444444', markeredgewidth=0.8))
    for patch, c in zip(ax.patches, colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)

    config.apply_ieee_theme(ax)
    ax.set_ylabel("Metric Score", fontsize=config.IEEE_LABEL_SIZE, fontname='Arial')
    ax.set_xlabel("")
    ax.set_ylim(30, 105)
    ax.set_xlim(-0.55, 4.85)
    ax.set_xticks(sorted(pos_map.values()))
    ax.set_xticklabels(['Pairwise\nAcc (%)', 'Macro\nF1 (x100)', 'Trial\nAUC (x100)',
                        'Seq Top-1\nAcc (%)', 'Macro\nF1 (x100)', 'AUC (x100)'],
                       fontsize=6.5, fontname='Arial')
    # Task group labels under the tick labels
    ax.text(0.7, -0.24, "P300 Speller Task", transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=7.5, fontname='Arial', fontweight='bold')
    ax.text(3.6, -0.24, "RSVP Task", transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=7.5, fontname='Arial', fontweight='bold')

    from matplotlib.patches import Patch
    legend = ax.legend(handles=[Patch(facecolor=metric_colors[m], alpha=0.85, label=m)
                                for m in ['Accuracy', 'Macro F1', 'AUC']],
                       frameon=True, facecolor='white', edgecolor='#e0e0e0',
                       prop={'size': config.IEEE_LEGEND_SIZE, 'family': 'Arial'},
                       title="Metric", title_fontsize=config.IEEE_LEGEND_SIZE)
    legend.get_title().set_fontname('Arial')

    plt.tight_layout()
    save_pdf = os.path.join(config.PLOT_DIR_GIST, "gist_ours_box_metrics.pdf")
    save_png = os.path.join(config.PLOT_DIR_GIST, "gist_ours_box_metrics.png")
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")

if __name__ == "__main__":
    generate_plot()
