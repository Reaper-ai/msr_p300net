import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config
from bciaut import load_itr_data as load_itr_bciaut
from gist import load_itr_data as load_itr_gist

DISPLAY = {
    'MSR_P300NET (Ours)': 'MSR-P300Net (Ours)',
    'EEGNetv4': 'EEGNet',
    'P3CNET': 'P3CNet',
    'EEGInception': 'EEG-Inception',
}

BCIAUT_MODELS = [("MSR_P300NET (Ours)", "msr_p300net"),
                 ("EEGNetv4", "eegnet"),
                 ("P3CNET", "p3cnet")]
GIST_MODELS = [("MSR_P300NET (Ours)", "msr_p300net"),
               ("EEGNetv4", "eegnet"),
               ("EEGInception", "eeginception")]


def generate_plot():
    bciaut_dfs = [load_itr_bciaut(name, folder) for name, folder in BCIAUT_MODELS]
    gist_dfs = [load_itr_gist(name, folder) for name, folder in GIST_MODELS]
    bciaut_dfs = [d for d in bciaut_dfs if d is not None]
    gist_dfs = [d for d in gist_dfs if d is not None]

    if not bciaut_dfs or not gist_dfs:
        print("Warning: Missing ITR results. Skipping combined ITR lines.")
        return

    # 30% shorter along y than the previous 3.9in version
    fig, axes = plt.subplots(2, 2, figsize=(config.IEEE_SINGLE_COL_WIDTH, 2.75))
    (ax_bc_acc, ax_gist_acc), (ax_bc_itr, ax_gist_itr) = axes

    # Zoomed y-limits (data crowds near the top) so model curves separate:
    # acc starts at 50%, BCIAUT ITR at 1.0, GIST ITR unchanged
    panels = [
        (ax_bc_acc, bciaut_dfs, 'Sequences', 'Accuracy (%)', "Accuracy (%)", [1, 5, 10], (50, 103), (50, 75, 100)),
        (ax_gist_acc, gist_dfs, 'Sequence', 'Accuracy (%)', "Accuracy (%)", [1, 7, 15], (50, 103), (50, 75, 100)),
        (ax_bc_itr, bciaut_dfs, 'Sequences', 'Bits/Selection', "ITR (bits/selection)", [1, 5, 10], (1.0, 3.25), (1, 2, 3)),
        (ax_gist_itr, gist_dfs, 'Sequence', 'ITR (bits/select)', "ITR (bits/selection)", [1, 7, 15], (-0.05, 1.1), (0, 0.5, 1.0)),
    ]

    for ax, dfs, x_col, y_col, y_label, xticks, ylim, yticks in panels:
        for df in dfs:
            model = df['Model'].iloc[0]
            ours = 'Ours' in model
            ax.plot(df[x_col], df[y_col],
                    color=config.MODEL_COLORS[model],
                    linewidth=1.5 if ours else 1.1,
                    alpha=1.0 if ours else 0.9,
                    marker='o' if ours else None,
                    markersize=2.6, markeredgewidth=0,
                    zorder=3 if ours else 2)
        config.apply_ieee_theme(ax)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Sequence Repetitions", fontsize=config.IEEE_LABEL_SIZE - 1.5,
                      fontname='Arial', labelpad=1.5)
        ax.set_ylabel(y_label, fontsize=config.IEEE_LABEL_SIZE - 1.5, fontname='Arial', labelpad=1.5)

    # Panel tags
    for ax, tag in [(ax_bc_acc, "(a) BCIAUT Acc"), (ax_gist_acc, "(b) GIST Acc"),
                    (ax_bc_itr, "(c) BCIAUT ITR"), (ax_gist_itr, "(d) GIST ITR")]:
        ax.text(0.04, 0.94, tag, transform=ax.transAxes, fontsize=7.5,
                fontname='Arial', fontweight='bold', va='top', ha='left')

    # Shared legend above the grid
    seen, handles = set(), []
    for model in ['MSR_P300NET (Ours)', 'EEGNetv4', 'P3CNET', 'EEGInception']:
        handles.append(Line2D([], [], color=config.MODEL_COLORS[model], linewidth=1.5,
                              label=DISPLAY[model]))
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
               fontsize=6.0, prop={'size': 6.0, 'family': 'Arial'},
               bbox_to_anchor=(0.5, 1.04), columnspacing=1.0, handlelength=1.6,
               handletextpad=0.4)

    plt.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.9, rect=[0, 0, 1, 0.93])
    save_pdf = os.path.join(config.PLOT_DIR_COMBINED, "combined_itr_lines.pdf")
    save_png = os.path.join(config.PLOT_DIR_COMBINED, "combined_itr_lines.png")
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")

if __name__ == "__main__":
    generate_plot()
