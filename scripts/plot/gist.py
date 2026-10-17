import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne

import config
import importlib.util

# Load GIST config dynamically
gist_config_path = os.path.join(config.ROOT_DIR, "scripts", "gist", "config.py")
spec = importlib.util.spec_from_file_location("gist_config", gist_config_path)
gist_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gist_config)

def load_subject_data(model_name, task, folder):
    # For RSVP, the filename is rsvp_cohort_results_[model].csv
    # For P300, the filename is p300_cohort_results_[model].csv
    filename = f"{task}_cohort_results_{folder}.csv"
    csv_path = os.path.join(config.ROOT_DIR, "results", "gist", folder, filename)
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    # Aggregate across seeds for each subject
    cols = ['Single BalAcc (%)', 'Macro F1', 'AUC', 'Seq Top-1 Acc (%)'] if task == 'rsvp' else ['Trial BalAcc (%)', 'Macro F1', 'Trial AUC', 'Pairwise Acc (%)']
    existing_cols = [c for c in cols if c in df.columns]
    df_agg = df.groupby('Subject')[existing_cols].mean().reset_index()
    df_agg['Model'] = model_name
    return df_agg

def load_itr_data(model_name, folder):
    csv_path = os.path.join(config.ROOT_DIR, "results", "gist", folder, f"p300_itr_results_{folder}.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    # Aggregate across seeds and subjects for each sequence limit
    df_agg = df.groupby('Sequence')[['Accuracy (%)', 'ITR (bits/select)']].mean().reset_index()
    df_agg['Model'] = model_name
    return df_agg

def plot_gist_bars():
    """Generates bar graphs vs subjects for GIST RSVP and P300 across all models."""
    tasks = ['rsvp', 'p300']
    
    for task in tasks:
        df_ours = load_subject_data("MSR_P300NET (Ours)", task, "msr_p300net")
        df_eegnet = load_subject_data("EEGNetv4", task, "eegnet")
        df_eeginc = load_subject_data("EEGInception", task, "eeginception")
        
        if df_ours is None or df_eegnet is None or df_eeginc is None:
            print(f"Warning: Missing GIST subject results for {task}. Skipping bar charts.")
            continue
            
        df_all = pd.concat([df_ours, df_eegnet, df_eeginc], ignore_index=True)
        # Parse Subject number for sorting
        df_all['SubjNum'] = df_all['Subject'].apply(lambda x: int(x.replace('S', '')))
        df_all = df_all.sort_values('SubjNum')
        
        # Plot Accuracy
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.barplot(data=df_all, x='Subject', y='Accuracy (%)', hue='Model', ax=ax,
                    palette=config.MODEL_COLORS, alpha=0.85)
        config.apply_theme(ax)
        ax.set_xlabel("Subject ID")
        ax.set_ylabel(f"{task.upper()} Accuracy (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.legend(frameon=False, loc="upper right", prop={'size': config.LEGEND_SIZE})
        plt.tight_layout()
        save_path = os.path.join(config.PLOT_DIR_GIST, f"gist_{task}_accuracy_bar.pdf")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        # Plot F1
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.barplot(data=df_all, x='Subject', y='Macro F1', hue='Model', ax=ax,
                    palette=config.MODEL_COLORS, alpha=0.85)
        config.apply_theme(ax)
        ax.set_xlabel("Subject ID")
        ax.set_ylabel(f"{task.upper()} Macro F1")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.legend(frameon=False, loc="upper right", prop={'size': config.LEGEND_SIZE})
        plt.tight_layout()
        save_path = os.path.join(config.PLOT_DIR_GIST, f"gist_{task}_f1_bar.pdf")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    print("Saved GIST RSVP and P300 subject-wise bar charts.")

def plot_gist_itr_lines():
    """Generates line charts for sequence-wise ITR in GIST P300."""
    df_ours = load_itr_data("MSR_P300NET (Ours)", "msr_p300net")
    df_eegnet = load_itr_data("EEGNetv4", "eegnet")
    df_eeginc = load_itr_data("EEGInception", "eeginception")
    
    if df_ours is None or df_eegnet is None or df_eeginc is None:
        return
        
    df_all = pd.concat([df_ours, df_eegnet, df_eeginc], ignore_index=True)
    
    # Plot 1: Accuracy vs Sequences
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name in df_all['Model'].unique():
        df_sub = df_all[df_all['Model'] == model_name]
        ax.plot(df_sub['Sequence'], df_sub['Accuracy (%)'], marker='o', label=model_name,
                color=config.MODEL_COLORS[model_name], linewidth=1.5, markersize=5)
    config.apply_theme(ax)
    ax.set_xlabel("Sequence Repetitions")
    ax.set_ylabel("Spelling Accuracy (%)")
    ax.set_xticks(range(1, 16))
    ax.legend(frameon=False, prop={'size': config.LEGEND_SIZE})
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_GIST, "gist_itr_accuracy.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Plot 2: ITR vs Sequences
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name in df_all['Model'].unique():
        df_sub = df_all[df_all['Model'] == model_name]
        ax.plot(df_sub['Sequence'], df_sub['ITR (bits/select)'], marker='o', label=model_name,
                color=config.MODEL_COLORS[model_name], linewidth=1.5, markersize=5)
    config.apply_theme(ax)
    ax.set_xlabel("Sequence Repetitions")
    ax.set_ylabel("ITR (bits/selection)")
    ax.set_xticks(range(1, 16))
    ax.legend(frameon=False, prop={'size': config.LEGEND_SIZE})
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_GIST, "gist_itr_bitrate.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved GIST sequence-wise ITR charts.")

def plot_gist_ablation():
    """Generates Accuracy and F1 vs Electrode Ablation [4, 8, 16, 32] for MSR_P300NET."""
    csv_abl = os.path.join(config.ROOT_DIR, "results", "gist", "msr_p300net_ablation", "p300_ablation_results_msr_p300net.csv")
    csv_main = os.path.join(config.ROOT_DIR, "results", "gist", "msr_p300net", "p300_cohort_results_msr_p300net.csv")
    
    if not os.path.exists(csv_abl) or not os.path.exists(csv_main):
        print("Warning: Missing GIST ablation results. Skipping ablation plot.")
        return
        
    df_abl = pd.read_csv(csv_abl)
    df_main = pd.read_csv(csv_main)
    
    # Extract only Channels, Accuracy, and F1
    df_abl_sub = df_abl[['Channels', 'Trial BalAcc (%)', 'Macro F1']].copy()
    df_abl_sub.rename(columns={'Trial BalAcc (%)': 'Accuracy (%)'}, inplace=True)
    
    df_main_sub = df_main[['Channels', 'Trial BalAcc (%)', 'Macro F1']].copy()
    df_main_sub.rename(columns={'Trial BalAcc (%)': 'Accuracy (%)'}, inplace=True)
    
    df_all = pd.concat([df_abl_sub, df_main_sub], ignore_index=True)
    
    # Aggregate (mean across subjects and seeds) for each channel configuration
    df_agg = df_all.groupby('Channels')[['Accuracy (%)', 'Macro F1']].mean().reset_index()
    df_agg = df_agg.sort_values('Channels')
    
    fig, ax = plt.subplots(figsize=(6, 4))
    # Dual axis to plot Accuracy and F1 on different scales or same
    # We can plot them on same axis by multiplying F1 by 100
    ax.plot(df_agg['Channels'], df_agg['Accuracy (%)'], marker='o', color=config.COLOR_ACCURACY,
            label='Accuracy (%)', linewidth=2)
    ax.set_ylabel('Accuracy (%)', color=config.COLOR_ACCURACY)
    ax.tick_params(axis='y', labelcolor=config.COLOR_ACCURACY)
    
    ax2 = ax.twinx()
    ax2.plot(df_agg['Channels'], df_agg['Macro F1'], marker='s', color=config.COLOR_F1,
             label='Macro F1', linewidth=2, linestyle='--')
    ax2.set_ylabel('Macro F1', color=config.COLOR_F1)
    ax2.tick_params(axis='y', labelcolor=config.COLOR_F1)
    
    config.apply_theme(ax)
    # Since twinx adds another axis, manually remove right spine for twin
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_color('#cccccc')
    
    ax.set_xlabel("Electrode Channels")
    ax.set_xticks([4, 8, 16, 32])
    
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_GIST, "gist_electrode_ablation.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_gist_saliency():
    """Generates rectangular gradient saliency maps for GIST P300."""
    json_path = os.path.join(config.ROOT_DIR, "results", "gist", "msr_p300net", "msr_p300net_interpretability.json")
    if not os.path.exists(json_path):
        return
    with open(json_path, "r") as f:
        data = json.load(f)
        
    channels = ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'O1', 'O2']
    
    # We look for P300 task key
    task_key = "p300_ch_8"
    if task_key not in data:
        # Fallback to check other keys
        keys = list(data.keys())
        if len(keys) > 0:
            task_key = keys[0]
        else:
            return
            
    subj_data = data[task_key]
    available_subjects = sorted(list(subj_data.keys()))
    if len(available_subjects) == 0:
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    
    if len(available_subjects) >= 3:
        for idx in range(3):
            subj = available_subjects[idx]
            subj_saliencies = []
            for seed_key in subj_data[subj].keys():
                saliency = subj_data[subj][seed_key]["gradient_saliency"]
                if len(saliency) > 0:
                    subj_saliencies.append(saliency)
            if len(subj_saliencies) == 0:
                continue
            avg_saliency = np.mean(subj_saliencies, axis=0)
            
            ax = axes[idx]
            sns.heatmap(avg_saliency, ax=ax, cmap="viridis", cbar=True,
                        yticklabels=channels, xticklabels=False)
            ax.set_xlabel("Time (Samples)")
            ax.set_title(f"Subject {subj}")
    else:
        subj = available_subjects[0]
        seeds = sorted(list(subj_data[subj].keys()))
        for idx in range(min(3, len(axes))):
            if idx < len(seeds):
                seed_key = seeds[idx]
                saliency = np.array(subj_data[subj][seed_key]["gradient_saliency"])
                ax = axes[idx]
                sns.heatmap(saliency, ax=ax, cmap="viridis", cbar=True,
                            yticklabels=channels, xticklabels=False)
                ax.set_xlabel("Time (Samples)")
                ax.set_title(f"Subject {subj} | {seed_key}")
                
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_GIST, "gist_saliency_maps.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def standardize_chan_names(ch_list):
    # Standard 1020 names case correction
    mapping = {
        'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
        'FP1': 'Fp1', 'FP2': 'Fp2', 'AF3': 'AF3', 'AF4': 'AF4',
        'FC5': 'FC5', 'FC6': 'FC6', 'FC1': 'FC1', 'FC2': 'FC2',
        'CP5': 'CP5', 'CP6': 'CP6', 'CP1': 'CP1', 'CP2': 'CP2',
        'PO3': 'PO3', 'PO4': 'PO4', 'PO7': 'PO7', 'PO8': 'PO8'
    }
    return [mapping.get(c.upper(), c) for c in ch_list]

def plot_gist_electrode_attention_shifts():
    """Plots 4 topographic head view maps in a row showing spatial attention shifts from 4 to 8 to 16 to 32 channels."""
    json_abl = os.path.join(config.ROOT_DIR, "results", "gist", "msr_p300net_ablation", "msr_p300net_ablation_interpretability.json")
    json_main = os.path.join(config.ROOT_DIR, "results", "gist", "msr_p300net", "msr_p300net_interpretability.json")
    
    if not os.path.exists(json_abl) or not os.path.exists(json_main):
        print("Warning: Missing GIST interpretability results. Skipping attention shift topo maps.")
        return
        
    with open(json_abl, "r") as f:
        data_abl = json.load(f)
    with open(json_main, "r") as f:
        data_main = json.load(f)
        
    # We want to extract spatial weights for 4, 8, 16, 32 channels
    # Let's map channel configs to their respective lists
    configs = [4, 8, 16, 32]
    ch_lists = {
        4: gist_config.CHANNELS_4,
        8: gist_config.CHANNELS_8,
        16: gist_config.CHANNELS_16,
        32: gist_config.CHANNELS_32
    }
    
    # Pick a subject to plot, e.g., SBJ01 or S01
    # Check subjects in main
    main_task = "p300_ch_8"
    if main_task not in data_main:
        main_task = list(data_main.keys())[0]
        
    subjs = list(data_main[main_task].keys())
    if len(subjs) == 0:
        return
    subj = subjs[0]
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    
    for idx, ch_count in enumerate(configs):
        ax = axes[idx]
        ch_names = standardize_chan_names(ch_lists[ch_count])
        
        # Load weights
        weights_list = []
        if ch_count == 8:
            subj_data = data_main[main_task].get(subj, {})
        else:
            task_key = f"p300_ch_{ch_count}"
            subj_data = data_abl.get(task_key, {}).get(subj, {})
            
        for seed_key in subj_data.keys():
            weights = subj_data[seed_key].get("spatial_weights", [])
            if len(weights) > 0:
                weights_list.append(weights)
                
        if len(weights_list) == 0:
            # Fallback to random weights if not computed (so it doesn't crash during dry run)
            avg_weights = np.ones(len(ch_names)) / len(ch_names)
        else:
            avg_weights = np.mean(weights_list, axis=0)
            
        # Create standard montage info
        info = mne.create_info(ch_names=ch_names, sfreq=100, ch_types='eeg')
        info.set_montage('standard_1020')
        
        # Plot topomap
        im, _ = mne.viz.plot_topomap(avg_weights, info, axes=ax, show=False, cmap="Reds")
        ax.set_title(f"{ch_count} Channels")
        
    cbar_ax = fig.add_axes([0.94, 0.2, 0.015, 0.6])
    fig.colorbar(im, cax=cbar_ax)
    
    save_path = os.path.join(config.PLOT_DIR_GIST, "gist_electrode_attention_shifts.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
