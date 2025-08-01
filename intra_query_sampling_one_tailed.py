"""Script to test estimators of individual query execution duration given samples within that execution.

Makes a nice chart to show that the 2a+c estimator is unbiased (well it would show that if it included the mean).

"""

from pathlib import Path
from pandas import DataFrame
from matplotlib import pyplot as pt
import seaborn as sns

true_query_length = 1000

results = []

for sampling_period in range(10, 1100, 25):
    for offset in range(0, min(true_query_length, sampling_period), 1):
        num_samples_in_query = (true_query_length - offset) // sampling_period
        last_sample_index = offset + sampling_period * num_samples_in_query
        min_length = last_sample_index
        max_length = min_length + sampling_period
        est_length_2s = last_sample_index * 2
        est_length_2s_plus = last_sample_index + offset
        est_length_h2n = last_sample_index + sampling_period / 2
        est_length_com = last_sample_index + min(sampling_period / 2, last_sample_index)
        results.append(
            (
                sampling_period,
                true_query_length / sampling_period,
                offset,
                true_query_length,
                num_samples_in_query,
                min_length,
                max_length,
                est_length_2s,
                est_length_2s_plus,
                est_length_h2n,
                est_length_com,
                est_length_2s if num_samples_in_query == 1 else est_length_h2n,
            )
        )


df = DataFrame(
    results,
    columns=(
        "Sampling Period / ms",
        "query_length_in_periods",
        "offset",
        "true_query_length",
        "num_samples_in_query",
        "min_length",
        "max_length",
        "est_length_2s",
        "Estimated Duration / ms",
        "est_length_h2n",
        "est_length_com",
        "est_length_com2",
    ),
)


chart_theme = "light"

if chart_theme == "dark":
    background = "#232425"
    inner_lines = "#a9a9b3"
    outer_lines = inner_lines
elif chart_theme == "light":
    inner_lines = "#232425"
    background = "white"
    outer_lines = "black"


chart_palette = ["#cfe2f3ff", "#d9ead3ff", "#d9d2e9ff", "#fff2ccff", "#f4ccccff"]
hue_params = {"palette": chart_palette}

sns.set_style(
    rc={
        "figure.facecolor": background,
        "axes.facecolor": background,
        "grid.color": inner_lines,
        "lines.color": inner_lines,
        "text.color": inner_lines,
        "xtick.color": outer_lines,
        "ytick.color": outer_lines,
        "axes.labelcolor": outer_lines,
    }
)

ax = pt.gca()  # Get current axes
ax.spines["top"].set_color(outer_lines)
ax.spines["bottom"].set_color(outer_lines)
ax.spines["left"].set_color(outer_lines)
ax.spines["right"].set_color(outer_lines)

sns.boxplot(
    df,
    # x="query_length_in_periods",
    x="Sampling Period / ms",
    y="Estimated Duration / ms",
    native_scale=True,
    whis=(5, 95),
    fliersize=0,
    color="#cfe2f3ff",
    linecolor=inner_lines,
    **hue_params,
)

fig = pt.gcf()
fig.set_size_inches(8, 4)

save_dir = Path(
    "/Users/simon/source/sonotley-dot-uk/content/posts/session-sampling-2/images"
)

if save_dir:
    pt.savefig(save_dir / f"2a-plus-c-{chart_theme}.svg")

pt.show()
