import os
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure plots folders exist
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PLOT_DIR_BCIAUT = os.path.join(ROOT_DIR, "plots", "bciaut")
PLOT_DIR_GIB = os.path.join(ROOT_DIR, "plots", "gib")
PLOT_DIR_GIST = os.path.join(ROOT_DIR, "plots", "gist")
PLOT_DIR_COMBINED = os.path.join(ROOT_DIR, "plots")

for d in [PLOT_DIR_BCIAUT, PLOT_DIR_GIB, PLOT_DIR_GIST, PLOT_DIR_COMBINED]:
    os.makedirs(d, exist_ok=True)

# Styling Config - IEEE Transactions Standards
FONT_FAMILY = "sans-serif"
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

# IEEE Dimensions (in inches)
IEEE_SINGLE_COL_WIDTH = 3.5
IEEE_DOUBLE_COL_WIDTH = 7.16

# IEEE Font Sizes (8-10 pt target)
IEEE_LABEL_SIZE = 9.0
IEEE_TITLE_SIZE = 9.5
IEEE_TICK_SIZE = 8.0
IEEE_LEGEND_SIZE = 8.0

# Legacy Large Text sizes for oversized figures
LABEL_SIZE = 26
TITLE_SIZE = 28
TICK_SIZE = 22
LEGEND_SIZE = 20

# Color Palette (consistent, professional IEEE colors)
COLOR_OUR_MODEL = "#0c2c56"      # Deep Navy for Ours
COLOR_EEGNET = "#ff7f0e"         # Orange for EEGNetv4
COLOR_P3CNET = "#2ca02c"         # Emerald Green for P3CNET
COLOR_EEGINCEPTION = "#17becf"   # Teal/Cyan for EEGInception

MODEL_COLORS = {
    'MSR_P300NET': COLOR_OUR_MODEL,
    'MSR_P300NET (Ours)': COLOR_OUR_MODEL,
    'HST_P300Net (Ours)': COLOR_OUR_MODEL,
    'Ours': COLOR_OUR_MODEL,
    'EEGNet': COLOR_EEGNET,
    'EEGNetv4': COLOR_EEGNET,
    'EEGInception': COLOR_EEGINCEPTION,
    'P3CNET': COLOR_P3CNET,
    'P3CNet': COLOR_P3CNET
}

COLOR_ACCURACY = "#0c2c56"  # Deep Navy for Accuracy
COLOR_F1 = "#d95f02"        # Rust/Coral for F1
COLOR_ITR = "#27ae60"

def apply_ieee_theme(ax):
    """Applies IEEE standard aesthetics (Arial/Helvetica, 8-9pt font, 0.8pt strokes)."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#222222')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_color('#222222')
    ax.spines['bottom'].set_linewidth(0.8)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='#d0d0d0', linewidth=0.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#d0d0d0', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=IEEE_TICK_SIZE, colors='#222222', length=3.5, width=0.8)
    ax.xaxis.label.set_size(IEEE_LABEL_SIZE)
    ax.yaxis.label.set_size(IEEE_LABEL_SIZE)
    ax.xaxis.label.set_color('#222222')
    ax.yaxis.label.set_color('#222222')
    if ax.get_title():
        ax.title.set_size(IEEE_TITLE_SIZE)
        ax.title.set_color('#111111')

def apply_theme(ax):
    """Applies clean paper-style aesthetics to a matplotlib axis."""
    apply_ieee_theme(ax)
