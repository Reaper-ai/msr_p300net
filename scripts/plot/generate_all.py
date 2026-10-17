import os
import sys

# Ensure local directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("="*65)
print("  BCI Research Paper Figure Generation Master Pipeline")
print("="*65)

# List of active plot generation modules
plots = [
    # IEEE Page-Wide Combined Figures
    ('combined_saliency_row', 'Page-Wide Saliency Maps Row (BCIAUT, GIB Healthy, GIB MS, GIB Other, GIST)'),
    ('combined_itr_lines', 'Combined ITR 2x2 (BCIAUT/GIST x Acc/ITR, Ours, Single Column)'),
    
    # BCIAUT Plots
    ('bciaut_box', 'BCIAUT Box Plot (Acc + F1 Combined, Pure Boxplot)'),
    ('bciaut_subject_lines', 'BCIAUT Subject-wise Accuracy Line Chart (Sessions 4-7, Our Model)'),
    ('bciaut_saliency', 'BCIAUT Saliency Maps (8 Electrodes: C3, Cz, C4, CPz, P3, Pz, P4, POz)'),
    ('bciaut_electrode_attention', 'BCIAUT Topographic Electrode Attention Map'),
    
    # GIB Plots
    ('gib_box', 'GIB Cohort Box Plot (All, Healthy, MS Only, Other Unhealthy - Our Model)'),
    ('gib_saliency', 'GIB Saliency Maps (8 Electrodes: Fz, Cz, Pz, Oz, P3, P4, PO7, PO8)'),
    ('gib_electrode_attention', 'GIB Topographic Electrode Attention Map (Healthy vs MS)'),
    
    # GIST Plots
    ('gist_box', 'GIST Box Plots (RSVP & P300 + Ours-only Combined)'),
    ('gist_saliency', 'GIST Saliency Maps (8 Electrodes: Fz, Cz, Pz, Oz, P3, P4, O1, O2)'),
    ('gist_electrode_attention', 'GIST Topographic Electrode Attention Map (4, 8, 16, 32 channels)')
]

for module_name, desc in plots:
    print(f"\nGenerating: {desc}...")
    try:
        # Dynamic import
        mod = __import__(module_name)
        if hasattr(mod, 'generate_plot'):
            mod.generate_plot()
        else:
            print(f"Error: Module '{module_name}' has no 'generate_plot' function.")
    except Exception as e:
        print(f"Failed to generate plot for '{module_name}': {e}")
        import traceback
        traceback.print_exc()

print("\n"+"="*65)
print("  All figure generation completed successfully.")
print("="*65)
