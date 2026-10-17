import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

def generate_plot():
    csv_path = os.path.join(config.ROOT_DIR, "results", "bciaut", "msr_p300net", "session_results.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: Missing {csv_path}. Skipping BCIAUT Subject Line Chart.")
        return
        
    df = pd.read_csv(csv_path)
    # Filter test sessions 4-7 and average across seeds for each subject & session
    df_test = df[df['Session'].isin([4, 5, 6, 7])].groupby(['Subject', 'Session'])['Obj Acc (%)'].mean().reset_index()
    df_test['SubjNum'] = df_test['Subject'].apply(lambda x: int(x.replace('SBJ', '')))
    df_test = df_test.sort_values(['Session', 'SubjNum'])
    
    # Generous width (4.8 inches) to ensure ample spacing between 2-digit subject labels (S10..S15)
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    
    session_styles = {
        4: {'label': 'Session 4', 'color': '#1f77b4', 'marker': 'o'},
        5: {'label': 'Session 5', 'color': '#ff7f0e', 'marker': 's'},
        6: {'label': 'Session 6', 'color': '#2ca02c', 'marker': '^'},
        7: {'label': 'Session 7', 'color': '#d62728', 'marker': 'D'}
    }
    
    for sess in [4, 5, 6, 7]:
        df_sess = df_test[df_test['Session'] == sess]
        x = df_sess['SubjNum'].values
        y = df_sess['Obj Acc (%)'].values
        style = session_styles[sess]
        
        ax.plot(x, y, marker=style['marker'], color=style['color'], label=style['label'],
                linewidth=1.2, markersize=4.2, alpha=0.9, zorder=3)
        
    config.apply_ieee_theme(ax)
    
    # Object decoding accuracy on Y-axis
    ax.set_ylabel("Object Decoding Accuracy (%)", fontsize=config.IEEE_LABEL_SIZE, fontname='Arial', labelpad=3)
    ax.set_xlabel("Subject", fontsize=config.IEEE_LABEL_SIZE, fontname='Arial', labelpad=3)
    ax.set_ylim(45, 104)
    ax.set_xlim(0.5, 15.5)
    
    # Subject ticks with generous spacing and padding
    ax.set_xticks(range(1, 16))
    ax.set_xticklabels([f"S{i:02d}" for i in range(1, 16)], fontsize=7.5, fontname='Arial')
    ax.tick_params(axis='x', pad=3.5, length=3.0, width=0.8)
    
    # Clean 2-column legend in lower-left area
    ax.legend(loc='lower left', ncol=2, frameon=True, facecolor='white', edgecolor='#e0e0e0',
              fontsize=7.5, prop={'size': 7.5, 'family': 'Arial'})

    plt.tight_layout()

    # Picture-in-picture: box plot of ours (Obj Acc + Macro F1 x100), subject-level, seed-averaged
    subj_csv = os.path.join(config.ROOT_DIR, "results", "bciaut", "msr_p300net", "subject_results.csv")
    if os.path.exists(subj_csv):
        df_subj = pd.read_csv(subj_csv).groupby('Subject')[['Obj Acc (%)', 'Macro F1']].mean().reset_index()
        df_inset = df_subj.melt(id_vars=['Subject'], value_vars=['Obj Acc (%)', 'Macro F1'],
                                var_name='Metric', value_name='Value')
        df_inset.loc[df_inset['Metric'] == 'Macro F1', 'Value'] *= 100.0
        df_inset['Metric'] = df_inset['Metric'].map({'Obj Acc (%)': 'Acc', 'Macro F1': 'F1'})

        # Placed in the empty band between S09 and S11 (all session lines sit above y~78 there)
        inset = ax.inset_axes([0.52, 0.07, 0.185, 0.36])
        sns.boxplot(data=df_inset, x='Metric', y='Value', order=['Acc', 'F1'], ax=inset,
                    color=config.COLOR_OUR_MODEL, width=0.5, showfliers=True,
                    boxprops=dict(alpha=0.9, linewidth=1.0),
                    medianprops=dict(linewidth=1.2, color='black'),
                    whiskerprops=dict(linewidth=1.0),
                    capprops=dict(linewidth=1.0))
        config.apply_ieee_theme(inset)
        inset.set_xlabel("")
        inset.set_ylabel("")
        inset.set_ylim(74, 103)
        inset.set_yticks([80, 90, 100])
        inset.tick_params(axis='both', labelsize=5.0, length=1.8, width=0.5, pad=1)
        for lbl in inset.get_xticklabels():
            lbl.set_fontname('Arial')
        inset.set_title("Ours: Acc / F1", fontsize=5.0, fontname='Arial', pad=1.5)
        inset.set_facecolor('white')
        inset.patch.set_alpha(1.0)
        inset.set_zorder(5)
        inset.xaxis.grid(False)
        for s in inset.spines.values():
            s.set_linewidth(0.5)
    
    save_pdf = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_subject_accuracy.pdf")
    save_png = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_subject_accuracy.png")
    plt.savefig(save_pdf, dpi=350, bbox_inches='tight')
    plt.savefig(save_png, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")

if __name__ == "__main__":
    generate_plot()
