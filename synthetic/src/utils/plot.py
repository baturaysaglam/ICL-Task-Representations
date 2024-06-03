import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats


# Constants for font size and line width
FONT_SIZE = 23
TICKS_FONT_SIZE_FRACTION = 0.9
TITLE_FRACTION = 1.25
LINEWIDTH = 2
TH_LINEWIDTH = 1.5
BORDER_LINEWIDTH = 1.5
COLORBAR_PAD = 30
INT_BORDER_LINEWIDTH_FRACTION = 0.5
SHADE_ALPHA = 0.15
HISTOGRAM_ALPHA = 0.4

# Set style
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
sns.set(context='paper', style="whitegrid", palette='colorblind')
sns.set_context("paper", rc={"lines.line_width":0.1})
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": False,
    "font.family":"Times New Roman",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.titlesize": 6,
    "axes.labelsize": 4,
    "font.size": 9,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 7,
    "xtick.labelsize": 4,
    "ytick.labelsize": 4,
    # reduce padding of x/y label ticks with plots
    "xtick.major.pad":0,
    "ytick.major.pad":0,
    # set figure dpi
    'figure.dpi': 600,
}
plt.rcParams.update(tex_fonts)


def create_heatmap(data, x_label='Layer index', y_label='Head index', cbar_title='Cosine similarity', title=None, dpi=100, save_dir=None, show=True):
    if show:
        time.sleep(0.1)

    # Create the heatmap using seaborn
    # sns.set(style='darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))  # You can adjust the figure size here
    heatmap = sns.heatmap(data, annot=False, cmap='coolwarm', cbar_kws={'label': cbar_title})

    # Setting the label sizes
    ax.set_xlabel(x_label, fontsize=FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE)

    # Setting tick sizes
    ax.tick_params(axis='x', labelsize=FONT_SIZE * TICKS_FONT_SIZE_FRACTION)
    ax.tick_params(axis='y', labelsize=FONT_SIZE * TICKS_FONT_SIZE_FRACTION)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONT_SIZE * TICKS_FONT_SIZE_FRACTION)
    cbar.set_label(cbar_title, fontsize=FONT_SIZE, rotation=270, labelpad=COLORBAR_PAD)  # Rotated colorbar title

    # Adding gridlines for cells
    ax.hlines(range(data.shape[0]), *ax.get_xlim(), colors='white', linewidth=BORDER_LINEWIDTH * INT_BORDER_LINEWIDTH_FRACTION)
    ax.vlines(range(data.shape[1]), *ax.get_ylim(), colors='white', linewidth=BORDER_LINEWIDTH * INT_BORDER_LINEWIDTH_FRACTION)

    # Reverse the y-axis to have the y-labels in ascending order from bottom to top
    ax.invert_yaxis()

    if title is not None:
        plt.title(title)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(BORDER_LINEWIDTH)

    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir, dpi=dpi, format='png')

    if show:
        plt.show()

    plt.clf()  # Clear the current figure after saving or showing


def plot_transformer(losses,
                     legends,
                     title,
                     colors=None,
                     ci_widths=None,
                     x_label="# in-context examples",
                     y_label="Mean squared error",
                     baseline=None,
                     save_path=None,
                     y_ticks_max=None,
                     y_ticks_interval=2.5,
                     dpi=600,
                     show=True):
    for i, (loss, legend) in enumerate(zip(losses, legends)):
        color = colors[i] if colors is not None else None
        line = plt.plot(loss, lw=2, label=legend, color=color)
        shade_color = color if colors is not None else line[0].get_color()

        # Adding the confidence interval shade
        if ci_widths is not None:
            lower_bound = np.array(loss) - ci_widths[i]
            upper_bound = np.array(loss) + ci_widths[i]
            plt.fill_between(range(len(loss)), lower_bound, upper_bound, color=shade_color, alpha=SHADE_ALPHA)

    if baseline is not None:
        plt.axhline(baseline, ls="--", color="gray", label="zero estimator")

    if y_ticks_max is None:
        y_ticks_max = 1.5 * max(loss.max() for loss in losses)

    length = len(losses[0])

    tick_positions = [0, length * 0.25, length * 0.5, length * 0.75, length - 1]
    tick_labels = ['0', '50', '100', '150', '200'] if length > 100 else ['0', '25', '50', '75', '100']

    plt.xticks(tick_positions, tick_labels)
    plt.yticks(np.arange(0, y_ticks_max + y_ticks_interval, y_ticks_interval))

    plt.xticks(fontsize=FONT_SIZE * TICKS_FONT_SIZE_FRACTION)
    plt.yticks(fontsize=FONT_SIZE * TICKS_FONT_SIZE_FRACTION)
    plt.xlabel(x_label, fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)

    x_position = 40 if length < 100 else 100

    plt.axvline(x=x_position, linestyle='--', linewidth=TH_LINEWIDTH, color='black')

    plt.title(title, fontsize=FONT_SIZE * TITLE_FRACTION, style='italic')
    # plt.legend(fontsize=FONT_SIZE * 0.9)

    plt.gca().spines['top'].set_linewidth(BORDER_LINEWIDTH)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_linewidth(BORDER_LINEWIDTH)
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_linewidth(BORDER_LINEWIDTH)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_linewidth(BORDER_LINEWIDTH)
    plt.gca().spines['right'].set_color('black')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format='pdf', dpi=dpi)

    if show:
        plt.show()

    plt.clf()  # Clear the current figure after saving or showing


def plot_histogram(data_1, data_2, title, colors=None, x_label="Value (1e-3)",
                   y_label="Frequency (1e2)", save_path=None, n_bins=150, dpi=150, show=True):
    if colors is None:
        colors = ['#FF5733', '#3357FF']  # Brighter and more saturated colors

    # # Enhanced alpha for better color visibility while allowing overlay visibility
    plt.hist(data_2, bins=n_bins, alpha=HISTOGRAM_ALPHA, color=colors[1], histtype='stepfilled')
    plt.hist(data_1, bins=n_bins, alpha=HISTOGRAM_ALPHA, color=colors[0], histtype='stepfilled')

    # Adjusting tick labels on the x-axis to ensure three ticks are always displayed
    x_min, x_max = min(min(data_1), min(data_2)), max(max(data_1), max(data_2))
    # x_min, x_max = 4.8, 6.8
    tick_positions = np.linspace(x_min, x_max, 5)  # 5 ticks to include edges and 3 in the middle
    tick_labels = [f"{value*1000:.2f}" for value in tick_positions]
    plt.xticks(tick_positions, tick_labels, fontsize=FONT_SIZE * TICKS_FONT_SIZE_FRACTION)

    # Setting the y-ticks if specified
    y_max = max(max(np.histogram(data_1, bins=n_bins)[0]), max(np.histogram(data_2, bins=n_bins)[0]))
    y_tick_positions = np.linspace(0, y_max, 6)  # 6 ticks including the bottom one
    y_tick_labels = [f"{value/100:.2f}" for value in y_tick_positions]
    plt.yticks(y_tick_positions, y_tick_labels, fontsize=FONT_SIZE * TICKS_FONT_SIZE_FRACTION)

    plt.xlabel(x_label, fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)

    if title is not None:
        plt.title(title, fontsize=16)

    plt.legend()

    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['right'].set_color('black')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format='pdf', dpi=dpi)

    if show:
        plt.show()

    plt.clf()


def compute_confidence_interval(losses, confidence=0.95):
    mean_rewards = np.mean(losses, axis=1)
    std_dev = np.std(mean_rewards, ddof=1)

    sem = std_dev / np.sqrt(len(mean_rewards))
    df = len(mean_rewards) - 1

    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    ci_width = t_critical * sem

    return ci_width
