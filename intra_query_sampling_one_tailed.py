"""Trying to estimate the total query time from the known true start and samples within it.

Technically if we know the true start of the next sampled state we could use that as well to correct a bit more.
In most cases it is probably of no importance... also IIRC EWS doesn't save samples of idle sessions, that's fine for
regular samples, but harder for jittered. Although given EWS doesn't do jitter it's pretty academic.

"""

from pandas import DataFrame
from matplotlib import pyplot as pt
import seaborn as sns

true_query_length = 1000

results = []

for sampling_period in range(10, 1500, 10):
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
        "sampling_period",
        "query_length_in_periods",
        "offset",
        "true_query_length",
        "num_samples_in_query",
        "min_length",
        "max_length",
        "est_length_2s",
        "est_length_2s_plus",
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
    x="sampling_period",
    y="est_length_2s_plus",
    native_scale=True,
    whis=(5, 95),
    fliersize=0,
    color="#cfe2f3ff",
    linecolor=inner_lines,
    **hue_params,
)

fig = pt.gcf()
fig.set_size_inches(8, 4)

pt.show()
