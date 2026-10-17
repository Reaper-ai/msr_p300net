import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne

import config

def load_subject_data(model_name, folder):
    csv_path = os.path.join(config.ROOT_DIR, "results", "bciaut", folder, "subject_results.csv")
    if not os.path.exists(csv_path):
        # Fallback to base folder if needed
        csv_path = os.path.join(config.ROOT_DIR, "results", "bciaut", f"{folder}_subject_results.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    # Aggregate across seeds
    df_agg = df.groupby('Subject')[['Obj Acc (%)', 'Macro F1']].mean().reset_index()
    df_agg['Model'] = model_name
    return df_agg

def load_session_data(model_name, folder):
    csv_path = os.path.join(config.ROOT_DIR, "results", "bciaut", folder, "session_results.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(config.ROOT_DIR, "results", "bciaut", f"{folder}_session_results.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    # Aggregate across seeds and subjects for sessions 4-7
    df_sub = df[df['Session'].isin([4, 5, 6, 7])]
    df_agg = df_sub.groupby(['Session'])[['Obj Acc (%)', 'Macro F1']].mean().reset_index()
    df_agg['Model'] = model_name
    return df_agg

def load_itr_data(model_name, folder):
    csv_path = os.path.join(config.ROOT_DIR, "results", "bciaut", folder, "itr_results.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(config.ROOT_DIR, "results", "bciaut", f"{folder}_itr_results.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    # Aggregate across seeds and subjects for each sequence limit
    df_agg = df.groupby('Sequences')[['Accuracy (%)', 'Bits/Selection']].mean().reset_index()
    df_agg['Model'] = model_name
    return df_agg

def plot_bciaut_box():
    """Generates combined Box Plot of Acc + F1 for all BCIAUT models with subject points."""
    df_ours = load_subject_data("MSR_P300NET (Ours)", "normal_hst")
    df_eegnet = load_subject_data("EEGNetv4", "eegnet")
    df_p3cnet = load_subject_data("P3CNET", "p3cnet")
    
    if df_ours is None or df_eegnet is None or df_p3cnet is None:
        print("Warning: Missing BCIAUT subject results. Skipping Box Plot.")
        return
        
    df_all = pd.concat([df_ours, df_eegnet, df_p3cnet], ignore_index=True)
    
    # We want a combined box plot: Y is Acc / F1 (x100), X is Metric (Acc / F1)
    df_melt = df_all.melt(id_vars=['Subject', 'Model'], value_vars=['Obj Acc (%)', 'Macro F1'],
                          var_name='Metric', value_name='Value')
    # Scale F1 to 0-100 range
    df_melt.loc[df_melt['Metric'] == 'Macro F1', 'Value'] *= 100.0
    df_melt['Metric'] = df_melt['Metric'].map({'Obj Acc (%)': 'Accuracy (%)', 'Macro F1': 'Macro F1 (x100)'})
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Transparent boxes
    sns.boxplot(data=df_melt, x='Metric', y='Value', hue='Model', ax=ax,
                palette=config.MODEL_COLORS, width=0.6, showfliers=False,
                boxprops=dict(alpha=0.6))
                
    # Subject points
    sns.stripplot(data=df_melt, x='Metric', y='Value', hue='Model', ax=ax,
                  palette=config.MODEL_COLORS, dodge=True, alpha=0.8,
                  linewidth=0.5, edgecolor='gray', size=4, legend=False)
                  
    config.apply_theme(ax)
    ax.set_ylabel("Performance Metric Score")
    ax.set_xlabel("")
    ax.set_ylim(40, 105)
    
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_box_metrics.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_bciaut_subject_lines():
    """Generates two separate line plots (Acc and F1) vs Subjects (1-15) for all models."""
    df_ours = load_subject_data("MSR_P300NET (Ours)", "normal_hst")
    df_eegnet = load_subject_data("EEGNetv4", "eegnet")
    df_p3cnet = load_subject_data("P3CNET", "p3cnet")
    
    if df_ours is None or df_eegnet is None or df_p3cnet is None:
        return
        
    df_all = pd.concat([df_ours, df_eegnet, df_p3cnet], ignore_index=True)
    # Parse Subject number for sorting
    df_all['SubjNum'] = df_all['Subject'].apply(lambda x: int(x.replace('SBJ', '')))
    df_all = df_all.sort_values('SubjNum')
    
    # Plot 1: Accuracy vs Subject
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name in df_all['Model'].unique():
        df_sub = df_all[df_all['Model'] == model_name]
        ax.plot(df_sub['SubjNum'], df_sub['Obj Acc (%)'], marker='o', label=model_name,
                color=config.MODEL_COLORS[model_name], linewidth=1.5, markersize=4)
    config.apply_theme(ax)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(range(1, 16))
    ax.set_xticklabels([f"S{i:02d}" for i in range(1, 16)])
    ax.legend(frameon=False, prop={'size': config.LEGEND_SIZE})
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_subject_accuracy.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Plot 2: F1 vs Subject
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name in df_all['Model'].unique():
        df_sub = df_all[df_all['Model'] == model_name]
        ax.plot(df_sub['SubjNum'], df_sub['Macro F1'], marker='o', label=model_name,
                color=config.MODEL_COLORS[model_name], linewidth=1.5, markersize=4)
    config.apply_theme(ax)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Macro F1")
    ax.set_xticks(range(1, 16))
    ax.set_xticklabels([f"S{i:02d}" for i in range(1, 16)])
    ax.legend(frameon=False, prop={'size': config.LEGEND_SIZE})
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_subject_f1.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved BCIAUT subject-wise accuracy and F1 line charts.")

def plot_bciaut_session_lines():
    """Generates two separate line plots (Acc and F1) vs Sessions (4-7) for all models."""
    df_ours = load_session_data("MSR_P300NET (Ours)", "normal_hst")
    df_eegnet = load_session_data("EEGNetv4", "eegnet")
    df_p3cnet = load_session_data("P3CNET", "p3cnet")
    
    if df_ours is None or df_eegnet is None or df_p3cnet is None:
        return
        
    df_all = pd.concat([df_ours, df_eegnet, df_p3cnet], ignore_index=True)
    
    # Plot 1: Accuracy vs Session
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name in df_all['Model'].unique():
        df_sub = df_all[df_all['Model'] == model_name]
        ax.plot(df_sub['Session'], df_sub['Obj Acc (%)'], marker='s', label=model_name,
                color=config.MODEL_COLORS[model_name], linewidth=1.5, markersize=5)
    config.apply_theme(ax)
    ax.set_xlabel("Session")
    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_xticks([4, 5, 6, 7])
    ax.legend(frameon=False, prop={'size': config.LEGEND_SIZE})
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_session_accuracy.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Plot 2: F1 vs Session
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name in df_all['Model'].unique():
        df_sub = df_all[df_all['Model'] == model_name]
        ax.plot(df_sub['Session'], df_sub['Macro F1'], marker='s', label=model_name,
                color=config.MODEL_COLORS[model_name], linewidth=1.5, markersize=5)
    config.apply_theme(ax)
    ax.set_xlabel("Session")
    ax.set_ylabel("Mean Macro F1")
    ax.set_xticks([4, 5, 6, 7])
    ax.legend(frameon=False, prop={'size': config.LEGEND_SIZE})
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_session_f1.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved BCIAUT session-wise accuracy and F1 line charts.")

def plot_bciaut_itr_lines():
    """Generates line charts for sequence-wise ITR (Acc vs Sequences and Bit Rate vs Sequences)."""
    df_ours = load_itr_data("MSR_P300NET (Ours)", "normal_hst")
    df_eegnet = load_itr_data("EEGNetv4", "eegnet")
    df_p3cnet = load_itr_data("P3CNET", "p3cnet")
    
    if df_ours is None or df_eegnet is None or df_p3cnet is None:
        return
        
    df_all = pd.concat([df_ours, df_eegnet, df_p3cnet], ignore_index=True)
    
    # Plot 1: Accuracy vs Sequences
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name in df_all['Model'].unique():
        df_sub = df_all[df_all['Model'] == model_name]
        ax.plot(df_sub['Sequences'], df_sub['Accuracy (%)'], marker='^', label=model_name,
                color=config.MODEL_COLORS[model_name], linewidth=1.5, markersize=5)
    config.apply_theme(ax)
    ax.set_xlabel("Sequence Repetitions")
    ax.set_ylabel("Spelling Accuracy (%)")
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False, prop={'size': config.LEGEND_SIZE})
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_itr_accuracy.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Plot 2: Bits/Selection vs Sequences
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name in df_all['Model'].unique():
        df_sub = df_all[df_all['Model'] == model_name]
        ax.plot(df_sub['Sequences'], df_sub['Bits/Selection'], marker='^', label=model_name,
                color=config.MODEL_COLORS[model_name], linewidth=1.5, markersize=5)
    config.apply_theme(ax)
    ax.set_xlabel("Sequence Repetitions")
    ax.set_ylabel("ITR (bits/selection)")
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False, prop={'size': config.LEGEND_SIZE})
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_itr_bitrate.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved BCIAUT sequence-wise ITR charts.")

def plot_bciaut_saliency():
    """Generates a 1x3 grid of rectangular gradient saliency maps for three subjects."""
    json_path = os.path.join(config.ROOT_DIR, "results", "bciaut", "interpretability_results.json")
    if not os.path.exists(json_path):
        return
    with open(json_path, "r") as f:
        data = json.load(f)
        
    subjects = ["SBJ01", "SBJ02", "SBJ03"]
    channels = ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'PO7', 'PO8']
    
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    
    for idx, subj in enumerate(subjects):
        if subj not in data:
            continue
        # Average saliency across seeds
        subj_saliencies = []
        for seed_key in data[subj].keys():
            saliency = data[subj][seed_key]["gradient_saliency"]
            if len(saliency) > 0:
                subj_saliencies.append(saliency)
                
        if len(subj_saliencies) == 0:
            continue
        avg_saliency = np.mean(subj_saliencies, axis=0) # shape [8, 128] or similar
        
        ax = axes[idx]
        # We plot time (x) vs channels (y)
        sns.heatmap(avg_saliency, ax=ax, cmap="viridis", cbar=True,
                    yticklabels=channels, xticklabels=False)
        ax.set_xlabel("Time (Samples)")
        ax.set_title(f"Subject {subj}")
        
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_saliency_maps.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_bciaut_electrode_attention():
    """Generates three topographic head view maps representing spatial electrode attentions for SBJ01, SBJ02, SBJ03."""
    json_path = os.path.join(config.ROOT_DIR, "results", "bciaut", "interpretability_results.json")
    if not os.path.exists(json_path):
        return
    with open(json_path, "r") as f:
        data = json.load(f)
        
    subjects = ["SBJ01", "SBJ02", "SBJ03"]
    ch_names = ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'PO7', 'PO8']
    
    # Create info with standard 1020 montage positions
    info = mne.create_info(ch_names=ch_names, sfreq=100, ch_types='eeg')
    info.set_montage('standard_1020')
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    
    for idx, subj in enumerate(subjects):
        if subj not in data:
            continue
        subj_weights = []
        for seed_key in data[subj].keys():
            weights = data[subj][seed_key]["spatial_weights"]
            if len(weights) > 0:
                subj_weights.append(weights)
        if len(subj_weights) == 0:
            continue
        avg_weights = np.mean(subj_weights, axis=0)
        
        ax = axes[idx]
        # Plot topomap
        im, _ = mne.viz.plot_topomap(avg_weights, info, axes=ax, show=False, cmap="Reds")
        ax.set_title(f"Subject {subj}")
        
    # Single colorbar on the side
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    fig.colorbar(im, cax=cbar_ax)
    
    save_path = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_electrode_attention.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
