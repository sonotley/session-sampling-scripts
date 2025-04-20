import numpy as np
from matplotlib import pyplot as pt
import seaborn as sns
import math

from session_sampling_simulator import session_simulator as sim
import polars as pl
from scipy.stats import lognorm, gaussian_kde


def make_query(
    id: int, mean_duration: int, duration_spread: int, session_proportion: float
) -> sim.Query:
    return sim.Query(
        id=id,
        mean_duration=mean_duration,
        target_periodicity=int(mean_duration / session_proportion),
        wait_state_ratios={1: 1},
        duration_distribution="lognormal",
        duration_spread=duration_spread,
    )


def generate_lognorm(mean: float, sd: float, length: int) -> np.ndarray:
    mu = math.log(mean**2 / math.sqrt(mean**2 + sd**2))
    sigma = np.sqrt(math.log(1 + (sd**2 / mean**2)))

    htd_base = lognorm(s=sigma, scale=np.exp(mu))
    print(htd_base.mean(), htd_base.std())
    return htd_base.pdf(range(length))


# todo: this still has some out by one issues
# todo: this seems right when the mean duration is << sample period, but not quite right thereafter
# todo: I'm starting to suspect my actual sampling code may be at fault
def generate_obs_dist_from_true_dist(true_dist: np.ndarray, sample_period: int):
    dist = np.zeros(true_dist.size)
    for i, x in enumerate(true_dist):
        # i = j + 1

        max_possible_samples = i // sample_period

        # First the case where you get the maximum number of samples in
        p_case_max = (i % sample_period) / sample_period
        lower_limit = int(sample_period * max_possible_samples)
        upper_limit = int(lower_limit + 2 * (i % sample_period))
        dist[lower_limit:upper_limit] += x * p_case_max / (upper_limit - lower_limit)

        # Then the case where you get one less than the max
        if max_possible_samples > 1:
            p_case_min = 1 - p_case_max
            # I redid this maths so it makes more sense but the result is the same
            c = int(sample_period * (max_possible_samples - 1))
            min_a = int(i - (sample_period * max_possible_samples))  # todo: maybe minus 1 inside the bracket to be picky?
            max_a = sample_period
            lower_limit = 2 * min_a + c
            upper_limit = 2 * max_a + c
            dist[lower_limit:upper_limit] += (
                p_case_min * x / (upper_limit - lower_limit)
            )

    return dist / sum(dist)


DURATION = 3600000

queries = [
    make_query(
        id=i + 1, mean_duration=80, duration_spread=i * 10, session_proportion=0.16
    )
    for i in range(0, 5)
]

results = []
all_observed_queries_results = []
all_true_queries_results = []

for sample_period in [50]:
    # for sample_period in 1000 / np.linspace(start=1, stop=10, num=5):

    for run in range(1, 2):
        sess = sim.generate_session(queries=queries, window_duration=DURATION)

        sess = sess // 10  # remove the wait ID for now

        true_regions = (
            pl.DataFrame(
                sim.find_contiguous_regions(sess),
                orient="row",
                schema=["start", "end", "literal"],
            )
            .with_columns(QID=pl.col("literal") % 100)
            .with_columns((pl.col("end") - pl.col("start") + 1).alias("true_duration"))
            .filter(pl.col("QID") != 0)
        )

        true_regions.write_csv("true.csv")
        all_true_queries_results.append(true_regions)
        true_summary = true_regions.group_by("QID").agg(
            [
                pl.mean("true_duration").alias("true_mean"),
                pl.var("true_duration").alias("true_var"),
            ]
        )

        for phase in np.linspace(0, 0.99, 20):
            st = sim.get_sample_times(
                session_duration=DURATION, sample_period=sample_period, phase=phase, strategy=sim.SamplingStrategy.UNIFORM
            )

            sample = sess[st]

            sample_data = zip(
                st[:-1],
                sample[:-1],
                sample[:-1] % 100,
                sample[:-1] // 100,
                st[:-1]
                - (sample[:-1] // 100)
                + 0.5,  # Assume query starts at the start of the ms and sample is in the middle
                sample[1:] % 100,
            )

            headers = [
                "Timestamp",
                "Raw",
                "QID",
                "start",
                "Time since start",
                "Next QID",
            ]

            # Select only the observations where the QID was non-zero
            # i.e. where we've observed a query execution
            non_zero_samples = (
                pl.DataFrame(sample_data, schema=headers)
                .filter(pl.col("QID") != 0)
                .join(true_regions, on="start")
            )

            # For each sequence of observations of the same execution, calculate the various duration estimates and weights etc
            observed_queries = (
                non_zero_samples.group_by(["start", "QID", "end", "true_duration"])
                .agg(
                    [
                        pl.min("Time since start").alias("a"),
                        pl.min("Timestamp").alias("First"),
                        pl.max("Timestamp").alias("Last"),
                    ]
                )
                .with_columns((pl.col("Last") - pl.col("First")).alias("c"))
                .with_columns((2 * pl.col("a") + pl.col("c")).alias("estimate"))
                .with_columns(
                    (pl.max_horizontal(sample_period / pl.col("estimate"), 1)).alias(
                        "weight"
                    )
                )
                .with_columns(
                    (pl.col("estimate") * pl.col("weight")).alias("weighted_est")
                )
                .with_columns((pl.col("estimate").mean().over("QID")).alias("est_mean"))
                .with_columns((pl.col("estimate").var().over("QID")).alias("est_var"))
                .with_columns((4 * pl.col("a").var().over("QID")).alias("2a_var"))
                .with_columns((pl.col("a").mean().over("QID")).alias("a_mean"))
                # .with_columns(pl.lit(40).alias("a_mean"))  # fixme: hacked this to the correct value to see what happened - weirdly seems to make things worse so it seems my approach isn't great
                .with_columns(((4 / 3) * pl.col("a_mean") ** 2).alias("exp_var"))
                .with_columns(
                    (
                        pl.col("estimate")
                        + (
                            (
                                1
                                - np.sqrt(
                                    1
                                    - pl.min_horizontal(
                                        pl.col("exp_var") / pl.col("est_var"), 1
                                    )
                                )
                            )
                            * (pl.col("est_mean") - pl.col("estimate"))
                        )
                    ).alias("est_adj")
                )
                .with_columns(
                    (
                        pl.col("a")
                        + (
                            (
                                1
                                - np.sqrt(  # todo: this seems slightly better without the sqrt but I can't justify why
                                    1
                                    - pl.min_horizontal(
                                        pl.col("exp_var") / (pl.col("2a_var")), 1
                                    )
                                )
                            )
                            * (pl.col("a_mean") - pl.col("a"))
                        )
                    ).alias("a_adj")
                )
                .with_columns(
                    (2 * pl.col("a_adj") + pl.col("c")).alias("estimate_aadj")
                )
                .with_columns(
                    (pl.max_horizontal(sample_period / pl.col("est_adj"), 1)).alias(
                        "weight_adj"
                    )
                )
                .with_columns(
                    (
                        pl.max_horizontal(sample_period / pl.col("estimate_aadj"), 1)
                    ).alias("weight_aadj")
                )
                .with_columns(
                    (pl.col("estimate") * pl.col("weight_adj")).alias(
                        "weighted_est_adj"
                    )
                )
                .with_columns(
                    (pl.col("est_adj") * pl.col("weight_adj")).alias(
                        "weighted_est_adj2"
                    )
                )
                .with_columns(
                    (pl.col("estimate") * pl.col("weight_aadj")).alias(
                        "weighted_est_aadj"
                    )
                )
                .with_columns(
                    (pl.col("estimate_aadj") * pl.col("weight_aadj")).alias(
                        "weighted_est_aadj2"
                    )
                )
            )

            # observed_queries.write_csv('oq.csv')
            all_observed_queries_results.append(observed_queries)

            # Aggregate the data by QID by dividing the sum of the weighted means by the sum of the weights
            summary_by_qid = (
                observed_queries.group_by(["QID"])
                .agg(
                    [
                        pl.mean("true_duration").alias("mean_true_sampled_only"),
                        pl.var("true_duration").alias("var_true_sampled_only"),
                        pl.mean("estimate").alias("mean_est_unweighted"),
                        (pl.sum("weighted_est") / pl.sum("weight")).alias(
                            "mean_est_weighted"
                        ),
                        (pl.sum("weighted_est_adj") / pl.sum("weight_adj")).alias(
                            "mean_est_weighted_adj"
                        ),
                        (pl.sum("weighted_est_adj2") / pl.sum("weight_adj")).alias(
                            "mean_est_weighted_adj2"
                        ),
                        (pl.sum("weighted_est_aadj") / pl.sum("weight_aadj")).alias(
                            "mean_est_weighted_aadj"
                        ),
                        (pl.sum("weighted_est_aadj2") / pl.sum("weight_aadj")).alias(
                            "mean_est_weighted_aadj2"
                        ),
                    ]
                )
                .join(true_summary, on="QID")
                .with_columns(phase=phase, run=run, sample_period=sample_period)
                .with_columns(sample_frequency=1000 / sample_period)
            )

            # df4.write_csv("summary.csv")
            results.append(summary_by_qid)

combined_summary = (
    pl.concat(results)
    .with_columns(
        [
            (pl.col(x) - pl.col("true_mean")).alias(f"error_{x}")
            for x in (
                "mean_est_weighted_adj2",
                "mean_est_weighted_adj",
                "mean_est_weighted",
                "mean_est_unweighted",
                "mean_est_weighted_aadj",
                "mean_est_weighted_aadj2",
            )
        ]
    )
    .with_columns(
        [
            (pl.col(x) / pl.col("true_mean")).alias(f"rel{x}")
            for x in (
                "error_mean_est_weighted_adj2",
                "error_mean_est_weighted_adj",
                "error_mean_est_weighted",
                "error_mean_est_unweighted",
                "error_mean_est_weighted_aadj",
                "error_mean_est_weighted_aadj2",
            )
        ]
    )
)

combined_summary.write_csv("combined_summary.csv")

# todo: join back to a table of query info here so we can use it in charts... maybe, or could use the actual mean
normalized_combined_summary = combined_summary.unpivot(
    index=["QID", "phase", "run", "sample_period", "sample_frequency"]
)
normalized_combined_summary.write_csv("ncs.csv")

aggregated_summary = combined_summary.group_by(["QID", "sample_period"]).agg(
    pl.mean("mean_est_weighted_adj")
)
aggregated_summary.write_csv("aggregated_summary.csv")

# todo: note a plot split out by query id or mean duration actually hides some of the bias
# todo: although saying that those are suppose to represent different queries so I would want to split them in real life

fig, axes = pt.subplots(2, 2)
fig2, axes2 = pt.subplots(4, 5)

a = sns.boxplot(
    normalized_combined_summary.filter(pl.col("variable").str.contains("relerror")),
    # x="query_length_in_periods",
    x="variable",
    y="value",
    native_scale=False,
    whis=(5, 95),
    fliersize=0,
    # color="#cfe2f3ff",
    ax=axes[0, 0],
    # hue="QID"
    # linecolor=inner_lines,
    # **hue_params,
)

sns.boxplot(
    normalized_combined_summary.filter(
        pl.col("variable") == "relerror_mean_est_weighted_aadj2"
    ),
    # x="query_length_in_periods",
    x="QID",
    y="value",
    native_scale=False,
    whis=(5, 95),
    fliersize=0,
    # color="#cfe2f3ff",
    ax=axes[0, 1],
    # hue="QID"
    # linecolor=inner_lines,
    # **hue_params,
)

sns.boxplot(
    # normalized_combined_summary.filter(pl.col("variable")=="relerror_mean_est_weighted_adj"),
    normalized_combined_summary.filter(
        pl.col("variable") == "relerror_mean_est_weighted_aadj2"
    ),
    # x="query_length_in_periods",
    x="sample_frequency",
    y="value",
    native_scale=True,
    whis=(5, 95),
    fliersize=0,
    # color="#cfe2f3ff",
    hue="QID",
    ax=axes[1, 0],
    # linecolor=inner_lines,
    # **hue_params,
)

sns.boxplot(
    # normalized_combined_summary.filter(pl.col("variable")=="relerror_mean_est_weighted_adj"),
    normalized_combined_summary.filter(
        pl.col("variable").str.contains("relerror_mean_est")
    ),
    # x="query_length_in_periods",
    x="QID",
    y="value",
    native_scale=True,
    whis=(5, 95),
    fliersize=0,
    # color="#cfe2f3ff",
    hue="variable",
    ax=axes[1, 1],
    # linecolor=inner_lines,
    # **hue_params,
)


all_true_queries = pl.concat(all_true_queries_results)
all_observed_queries = pl.concat(all_observed_queries_results)
cp = sns.color_palette("hls", 8)
bw = 2
for i in range(1, 6):
    c = cp[i - 1]
    true_dist = generate_lognorm(
        sd=queries[i - 1].duration_spread + 0.01,  # add 0.01 just to convince zero case to work
        mean=queries[i - 1].mean_duration,
        length=500,
    )
    sns.histplot(
        all_true_queries.filter(pl.col("QID") == i),
        x="true_duration",
        ax=axes2[0, i - 1],
        binwidth=bw,
        binrange=(0, 500),
        color=c,
    )
    sns.lineplot(
        true_dist * len(all_true_queries.filter(pl.col("QID") == i)) * bw,
        ax=axes2[0, i - 1],
        color='grey',
        linewidth=1,
    )
    sns.histplot(
        all_observed_queries.filter(pl.col("QID") == i),
        x="true_duration",
        ax=axes2[1, i - 1],
        binwidth=bw,
        binrange=(0, 500),
        color=c,
    )
    sns.lineplot(
        true_dist * len(all_observed_queries.filter(pl.col("QID") == i)) * bw,
        ax=axes2[1, i - 1],
        color='grey',
        linestyle='dotted',
        linewidth=1,
    )
    exp_obs_dist = true_dist * [min(1, x/sample_period) for x in range(500)]
    exp_obs_dist /= sum(exp_obs_dist)
    sns.lineplot(
        exp_obs_dist * len(all_observed_queries.filter(pl.col("QID") == i)) * bw,
        ax=axes2[1, i - 1],
        color='grey',
        linewidth=1,
    )
    sns.histplot(
        all_observed_queries.filter(pl.col("QID") == i),
        x="estimate",
        ax=axes2[2, i - 1],
        binwidth=bw,
        binrange=(0, 500),
        color=c,
    )
    sns.lineplot(
        generate_obs_dist_from_true_dist(
            true_dist,
            sample_period=sample_period,
        )* len(all_observed_queries.filter(pl.col("QID") == i)) * bw,
        ax=axes2[2, i - 1],
        color='grey',
        linewidth=1,
    )
    # sns.histplot(
    #     all_observed_queries.filter(pl.col("QID") == i),
    #     x="estimate",
    #     weights="weight_aadj",
    #     ax=axes2[3, i - 1],
    #     binwidth=bw,
    #     binrange=(0, 500),
    #     color=c,
    # )
    # sns.histplot(
    #     all_observed_queries.filter(pl.col("QID") == i),
    #     x="estimate_aadj",
    #     weights="weight_aadj",
    #     ax=axes2[4, i - 1],
    #     binwidth=bw,
    #     binrange=(0, 500),
    #     color=c,
    # )
    sns.lineplot(
        generate_obs_dist_from_true_dist(
            true_dist,
            sample_period=sample_period,
        ),
        ax=axes2[3, i - 1],
        color=c,
    )
    raw_est = (
        all_observed_queries.filter(pl.col("QID") == i).select("estimate").to_numpy().T
    )
    kde = gaussian_kde(np.concat((raw_est, -raw_est), axis=1), bw_method=0.02)
    print(kde.factor)
    sns.lineplot(
        2 * kde.pdf(range(500)), ax=axes2[3, i - 1], color="grey", linewidth=1, #linestyle="dotted"
    )


# print(normalized_combined_summary.group_by(['QID', 'variable']).agg([
#             pl.mean("value").alias("mean"),
#             pl.std("value").alias("std"),
#         ]).filter(pl.col('variable').is_in('true_mean', 'mean_est_unweighted').sort(by='QID'))
#

a.tick_params(axis="x", labelrotation=90)

axes[0, 0].grid(True, axis="y")
axes[0, 1].grid(True, axis="y")
axes[1, 0].grid(True, axis="y")
axes[1, 1].grid(True, axis="y")
pt.show()


# Because we know the true duration of the sampled queries as well as the true duration of all queries we can see if there's any obvious relationship between the two
# Haven't found one yet... they do seem kind of related by the variance, but not in a very obvious way
