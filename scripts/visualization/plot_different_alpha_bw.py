import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Configure matplotlib to support Chinese characters
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Create Chinese font property for explicit use
try:
    chinese_font = fm.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
except (FileNotFoundError, RuntimeError):
    chinese_font = fm.FontProperties(family='WenQuanYi Zen Hei')

# Black and white style settings for different methods
method_styles = {
    "fedavg": {"linestyle": "-", "marker": "o", "hatch": "..", "color": "black"},
    "heterofl": {"linestyle": "--", "marker": "s", "hatch": "\\\\", "color": "black"},
    "fedrolex": {"linestyle": "-.", "marker": "^", "hatch": "//", "color": "black"},
    "fd": {"linestyle": ":", "marker": "d", "hatch": "++", "color": "black"},
    "fedecover": {"linestyle": "-", "marker": "*", "hatch": None, "color": "black"},  # Five-pointed star, solid black fill
}

# Grayscale colors for better distinction - FedEcover is darkest to highlight it
method_grays = {
    "fedavg": "0.4",      # Medium gray
    "heterofl": "0.6",    # Very light gray
    "fedrolex": "0.6",    # Light gray
    "fd": "0.2",          # Dark gray
    "fedecover": "0.0",   # Black (most prominent)
}

method_labels = {
    "fedavg": "FedAvg + GSD",
    "heterofl": "HeteroFL + GSD",
    "fedrolex": "FedRolex + GSD",
    "fd": "FD-m + GSD",
    "fedecover": "FedEcover",
    "fedavg-no-gsd": "FedAvg w/o GSD",
    "heterofl-no-gsd": "HeteroFL w/o GSD",
    "fedrolex-no-gsd": "FedRolex w/o GSD",
    "fd-no-gsd": "FD-m w/o GSD",
    "fedecover-no-gsd": "FedEcover w/o GSD",
}

sub_dir = "different-alpha"
fig_dir = "figures"
csv_dir = "results"
if sub_dir:
    csv_dir = os.path.join(csv_dir, sub_dir)
    fig_dir = os.path.join(fig_dir, sub_dir)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# Read data from csv files
accuracy_df = pd.read_csv(
    os.path.join(csv_dir, "different_alpha_accuracy_100clients.csv")
)
speedup_df = pd.read_csv(os.path.join(csv_dir, "different_alpha_speedup_100clients.csv"))

methods = accuracy_df["Method"].tolist()
alphas = accuracy_df.columns[1:].astype(float).tolist()

accuracy = {
    method: accuracy_df[accuracy_df["Method"] == method].values[0][1:].tolist()
    for method in methods
}
speedup = {
    method: speedup_df[speedup_df["Method"] == method].values[0][1:].tolist()
    for method in methods
}

# Set matplotlib to use black and white style
plt.style.use('default')
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['black'])

# Create a figure and set the x-axis label and y-axis label
fig, ax1 = plt.subplots()
ax1.set_xlabel("α", fontsize=20)
ax1.set_ylabel("准确率（%）", fontsize=20, fontproperties=chinese_font)

# Create a second y-axis for the speedup
ax2 = ax1.twinx()
ax2.set_ylabel("加速比", fontsize=20, fontproperties=chinese_font)

bar_width = 0.1
index = np.arange(len(alphas))

# Plot the accuracy line chart with different line styles and markers
for i, method in enumerate(methods):
    style = method_styles.get(method, {"linestyle": "-", "marker": "o", "color": "black"})
    gray_color = method_grays.get(method, "0.0")
    
    # Make FedEcover line thicker to highlight it
    line_width = 3 if method == "fedecover" else 2
    marker_size = 10 if method == "fedecover" else 8
    
    ax1.plot(
        index + bar_width * (len(methods) - 1) / 2,
        accuracy[method],
        marker=style["marker"],
        linestyle=style["linestyle"],
        label=f"{method_labels[method]}",
        color=gray_color,
        linewidth=line_width,
        markersize=marker_size,
        markerfacecolor='white',
        markeredgecolor=gray_color,
        markeredgewidth=2
    )

ax1.tick_params(axis="y")
ax1.set_xticks(
    index + bar_width * (len(methods) - 1) / 2
)  # Set the x-axis ticks to the midpoints of the bars in the bar chart.
ax1.set_xticklabels(
    alphas, fontdict={"fontsize": 16}
)  # Set the x-axis tick labels to alpha values.
ax1.set_yticklabels(ax1.get_yticks(), fontdict={"fontsize": 16})

# Plot a bar chart of speedup with different hatching patterns
bars = []
for i, method in enumerate(methods):
    style = method_styles.get(method, {"hatch": "", "color": "black"})
    gray_color = method_grays.get(method, "0.0")
    
    # For fedecover, use solid black fill
    bar_color = 'black' if method == "fedecover" else 'white'
    
    bar = ax2.bar(
        index + i * bar_width,
        speedup[method],
        bar_width,
        label=f"{method_labels[method]}",
        alpha=0.8,
        color=bar_color,
        edgecolor=gray_color,
        hatch=style["hatch"],
        linewidth=1.5
    )
    bars.append(bar)

ax2.tick_params(axis="y")

# Create separate legends for lines and bars
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Create legend elements for line chart (accuracy)
line_legend_elements = []
for i, method in enumerate(methods):
    style = method_styles.get(method, {"linestyle": "-", "marker": "o", "color": "black"})
    gray_color = method_grays.get(method, "0.0")
    
    # Make FedEcover marker bigger in legend too
    marker_size = 10 if method == "fedecover" else 8
    line_width = 3 if method == "fedecover" else 2
    
    line_element = Line2D([0], [0], 
                         color=gray_color, 
                         linestyle=style["linestyle"],
                         marker=style["marker"],
                         linewidth=line_width,
                         markersize=marker_size,
                         markerfacecolor='white',
                         markeredgecolor=gray_color,
                         markeredgewidth=2,
                         label=f"{method_labels[method]}")
    line_legend_elements.append(line_element)

# Create legend elements for bar chart (speedup)
bar_legend_elements = []
for i, method in enumerate(methods):
    style = method_styles.get(method, {"hatch": "", "color": "black"})
    gray_color = method_grays.get(method, "0.0")
    
    # Use black fill for fedecover in legend
    legend_facecolor = 'black' if method == "fedecover" else 'white'
    
    bar_element = Patch(facecolor=legend_facecolor, 
                       edgecolor=gray_color, 
                       hatch=style["hatch"],
                       linewidth=1.5,
                       label=f"{method_labels[method]}")
    bar_legend_elements.append(bar_element)

# Add section headers
# line_legend_elements.insert(0, Line2D([0], [0], color='none', label='准确率(线):'))
# bar_legend_elements.insert(0, Patch(facecolor='none', edgecolor='none', label='加速比(柱):'))

# Combine all legend elements
all_legend_elements = line_legend_elements + bar_legend_elements

# Create the combined legend
ax1.legend(handles=all_legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=10,
          columnspacing=1.5, handletextpad=0.5)

ax2.set_xticks(index + bar_width * (len(methods) - 1) / 2)
ax2.set_xticklabels(alphas, fontdict={"fontsize": 16})
ax2.set_yticklabels(ax2.get_yticks(), fontdict={"fontsize": 16})

# Set background to white and remove grid for cleaner black and white appearance
ax1.set_facecolor('white')
ax2.set_facecolor('white')
fig.patch.set_facecolor('white')

# Add subtle grid for better readability in black and white
ax1.grid(True, linestyle=':', alpha=0.3, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "different-alpha-100clients-bw.pdf"), 
           bbox_inches="tight", facecolor='white', edgecolor='none')
plt.savefig(os.path.join(fig_dir, "different-alpha-100clients-bw.png"), 
           bbox_inches="tight", facecolor='white', edgecolor='none', dpi=300)

print("Black and white plots saved to:")
print("- " + os.path.join(fig_dir, 'different-alpha-10clients-bw.pdf'))
print("- " + os.path.join(fig_dir, 'different-alpha-10clients-bw.png'))
