import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
import importlib.util

# Load GIB config dynamically
gib_config_path = os.path.join(config.ROOT_DIR, "scripts", "gib", "config.py")
spec = importlib.util.spec_from_file_location("gib_config", gib_config_path)
gib_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gib_config)

def load_subject_data(model_name, folder):
    csv_path = os.path.join(config.ROOT_DIR, "results", "gib", folder, "subject_results.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    # Aggregate across seeds for each subject
    df_agg = df.groupby('Subject_ID')[['Obj Acc (%)', 'Macro F1']].mean().reset_index()
    df_agg['Model'] = model_name
    return df_agg

def get_cohort(subj_id):
    meta = gib_config.SUBJECT_METADATA.get(subj_id, {'condition': 'Healthy'})
    return meta['condition']

def plot_gib_box():
    """Generates combined boxplot for GIB grouped by All, Healthy, and Unhealthy (Pathology) cohorts."""
    df_ours = load_subject_data("MSR_P300NET (Ours)", "msr_p300net")
    df_eeginc = load_subject_data("EEGInception", "eeginception")
    df_p3cnet = load_subject_data("P3CNET", "p3cnet")
    
    if df_ours is None or df_eeginc is None or df_p3cnet is None:
        print("Warning: Missing GIB subject results. Skipping Box Plot.")
        return
        
    df_all = pd.concat([df_ours, df_eeginc, df_p3cnet], ignore_index=True)
    df_all['Cohort'] = df_all['Subject_ID'].apply(get_cohort)
    
    # We want 3 groups: All, Healthy, Unhealthy
    # Let's create a duplicated dataframe for "All" cohort so we can plot it alongside Healthy and Unhealthy
    df_all_cohort = df_all.copy()
    df_all_cohort['Cohort'] = 'All'
    
    df_plot = pd.concat([df_all, df_all_cohort], ignore_index=True)
    
    # Melt for combined Acc and F1
    df_melt = df_plot.melt(id_vars=['Subject_ID', 'Model', 'Cohort'], value_vars=['Obj Acc (%)', 'Macro F1'],
                           var_name='Metric', value_name='Value')
    # Scale F1 to 0-100 range
    df_melt.loc[df_melt['Metric'] == 'Macro F1', 'Value'] *= 100.0
    df_melt['Metric'] = df_melt['Metric'].map({'Obj Acc (%)': 'Accuracy (%)', 'Macro F1': 'Macro F1 (x100)'})
    
    # Plot as a 1x3 grid of boxplots: All, Healthy, Unhealthy
    cohorts = ['All', 'Healthy', 'Unhealthy']
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    
    for idx, cohort in enumerate(cohorts):
        ax = axes[idx]
        df_sub = df_melt[df_melt['Cohort'] == cohort]
        sns.boxplot(data=df_sub, x='Metric', y='Value', hue='Model', ax=ax,
                    palette=config.MODEL_COLORS, width=0.6, showfliers=False,
                    boxprops=dict(alpha=0.6))
        config.apply_theme(ax)
        ax.set_title(f"Cohort: {cohort}")
        ax.set_xlabel("")
        ax.set_ylim(30, 105)
        if idx > 0:
            ax.set_ylabel("")
            ax.legend().remove()
        else:
            ax.set_ylabel("Performance Score")
            ax.legend(frameon=False, prop={'size': config.LEGEND_SIZE})
            
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_GIB, "gib_box_cohort_metrics.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_gib_bars():
    """Generates bar graphs comparing Acc and F1 vs Subjects (73 subjects) for all models."""
    df_ours = load_subject_data("MSR_P300NET (Ours)", "msr_p300net")
    df_eeginc = load_subject_data("EEGInception", "eeginception")
    df_p3cnet = load_subject_data("P3CNET", "p3cnet")
    
    if df_ours is None or df_eeginc is None or df_p3cnet is None:
        return
        
    df_all = pd.concat([df_ours, df_eeginc, df_p3cnet], ignore_index=True)
    df_all = df_all.sort_values('Subject_ID')
    
    # Plot 1: Accuracy vs Subject Bar Graph
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.barplot(data=df_all, x='Subject_ID', y='Obj Acc (%)', hue='Model', ax=ax,
                palette=config.MODEL_COLORS, alpha=0.85)
    config.apply_theme(ax)
    ax.set_xlabel("Subject ID (0-72)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.legend(frameon=False, loc="upper right", prop={'size': config.LEGEND_SIZE})
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_GIB, "gib_subject_accuracy_bar.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Plot 2: F1 vs Subject Bar Graph
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.barplot(data=df_all, x='Subject_ID', y='Macro F1', hue='Model', ax=ax,
                palette=config.MODEL_COLORS, alpha=0.85)
    config.apply_theme(ax)
    ax.set_xlabel("Subject ID (0-72)")
    ax.set_ylabel("Macro F1")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.legend(frameon=False, loc="upper right", prop={'size': config.LEGEND_SIZE})
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_GIB, "gib_subject_f1_bar.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved GIB subject-wise Accuracy and F1 bar charts.")

def plot_gib_saliency():
    """Generates rectangular gradient saliency maps for GIB."""
    json_path = os.path.join(config.ROOT_DIR, "results", "gib", "msr_p300net", "interpretability_results.json")
    if not os.path.exists(json_path):
        return
    with open(json_path, "r") as f:
        data = json.load(f)
        
    channels = ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'PO7', 'PO8']
    
    # Check what subjects are in the data
    available_subjects = sorted(list(data.keys()))
    if len(available_subjects) == 0:
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    
    if len(available_subjects) >= 3:
        # Plot first 3 subjects
        for idx in range(3):
            subj = available_subjects[idx]
            subj_saliencies = []
            for seed_key in data[subj].keys():
                saliency = data[subj][seed_key]["gradient_saliency"]
                if len(saliency) > 0:
                    subj_saliencies.append(saliency)
            if len(subj_saliencies) == 0:
                continue
            avg_saliency = np.mean(subj_saliencies, axis=0)
            # Remove trial dimensions if they exist
            if avg_saliency.ndim == 3:
                avg_saliency = avg_saliency.mean(axis=0)
            
            ax = axes[idx]
            sns.heatmap(avg_saliency, ax=ax, cmap="viridis", cbar=True,
                        yticklabels=channels, xticklabels=False)
            ax.set_xlabel("Time (Samples)")
            ax.set_title(f"Subject {subj}")
    else:
        # We only have 1 or 2 subjects. Let's plot different seeds for the first available subject
        subj = available_subjects[0]
        seeds = sorted(list(data[subj].keys()))
        for idx in range(min(3, len(axes))):
            if idx < len(seeds):
                seed_key = seeds[idx]
                saliency = np.array(data[subj][seed_key]["gradient_saliency"])
                if saliency.ndim == 3:
                    saliency = saliency.mean(axis=0)
                ax = axes[idx]
                sns.heatmap(saliency, ax=ax, cmap="viridis", cbar=True,
                            yticklabels=channels, xticklabels=False)
                ax.set_xlabel("Time (Samples)")
                ax.set_title(f"Subject {subj} | {seed_key}")
                
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR_GIB, "gib_saliency_maps.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
