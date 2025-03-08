import numpy as np
from matplotlib import pyplot as pt
import seaborn as sns

from session_sampling_simulator import session_simulator as sim
import polars as pl


def make_query(id: int, mean_duration: int, session_proportion: float) -> sim.Query:
    return sim.Query(
        id=id,
        mean_duration=mean_duration,
        target_periodicity=int(mean_duration / session_proportion),
        wait_state_ratios={1: 1},
        duration_distribution='exponential',  # my results seem somewhat biased when using uniform, especially when variance is small
        duration_spread=int(0)  # only used with uniform
    )


DURATION = 3600000

queries = [
    make_query(id=i, mean_duration=i * 40, session_proportion=0.16)
    for i in range(1, 6)
]

results = []

for sample_period in 1000/np.linspace(start=1, stop=10, num=5):
    for run in range(1, 5):
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
        true_summary = true_regions.group_by("QID").agg(
            [
                pl.mean("true_duration").alias("true_mean"),
                pl.var("true_duration").alias("true_var"),
            ]
        )

        for phase in np.linspace(0, 0.99, 10):
            st = sim.get_sample_times(
                session_duration=DURATION, sample_period=sample_period, phase=phase
            )

            sample = sess[st]

            query_elapsed_times = zip(
                st[:-1],
                sample[:-1],
                sample[:-1] % 100,
                sample[:-1] // 100,
                st[:-1]
                - (sample[:-1] // 100)
                + 1,  # Added one to fudge div0 errors, need to think about this
                sample[1:] % 100,
            )

            headers = ["Timestamp", "Raw", "QID", "start", "Time since start", "Next QID"]

            non_zero_samples = (
                pl.DataFrame(query_elapsed_times, schema=headers)
                .filter(pl.col("QID") != 0)
                .join(true_regions, on="start")
            )

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
                .with_columns((pl.col("estimate") * pl.col("weight")).alias("weighted_est"))
                .with_columns((pl.col("estimate").mean().over("QID")).alias("est_mean"))
                .with_columns((pl.col("estimate").var().over("QID")).alias("est_var"))
                .with_columns((4 * pl.col("a").var().over("QID")).alias("2a_var"))
                .with_columns((pl.col("a").mean().over("QID")).alias("a_mean"))
                .with_columns(((4/3) * pl.col("a_mean") ** 2).alias("exp_var"))
                # todo: might be better if we defined this as the proportion of variance in 2a that's explained and only applied it to the 2a part of d, trying this below
                .with_columns(
                    (
                            pl.col("estimate")
                            + (
                                ( 1 - np.sqrt(
                                        1 - pl.min_horizontal(pl.col("exp_var") / pl.col("est_var"), 1)
                                    ))
                                    * (pl.col("est_mean")-pl.col("estimate"))
                            )
                    ).alias("est_adj")
            )
                .with_columns(
                    (
                            pl.col("a")
                            + (
                                (1 - np.sqrt(
                                        1 - pl.min_horizontal(pl.col("exp_var") / (pl.col("2a_var")), 1)
                                    ))
                                    * (pl.col("a_mean") - pl.col("a"))
                            )
                    ).alias("a_adj")
                )
                .with_columns((2 * pl.col("a_adj") + pl.col("c")).alias("estimate_aadj"))

                .with_columns(
                    (pl.max_horizontal(sample_period / pl.col("est_adj"), 1)).alias(
                        "weight_adj"
                    )
                )
                .with_columns(
                    (pl.max_horizontal(sample_period / pl.col("estimate_aadj"), 1)).alias(
                        "weight_aadj"
                    )
                )
                .with_columns(
                    (pl.col("estimate") * pl.col("weight_adj")).alias("weighted_est_adj")
                )
                .with_columns(
                    (pl.col("est_adj") * pl.col("weight_adj")).alias("weighted_est_adj2"))
                .with_columns(
                    (pl.col("estimate") * pl.col("weight_aadj")).alias("weighted_est_aadj"))
            )

            observed_queries.write_csv('oq.csv')

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
                    ]
                )
                .join(true_summary, on="QID")
                .with_columns(phase=phase, run=run, sample_period=sample_period)
                .with_columns(sample_frequency=1000/sample_period)
            )

            # df4.write_csv("summary.csv")
            results.append(summary_by_qid)

combined_summary = pl.concat(results).with_columns(
    [
        (pl.col(x) - pl.col("true_mean")).alias(f"error_{x}")
        for x in ("mean_est_weighted_adj2", "mean_est_weighted_adj", "mean_est_weighted", "mean_est_unweighted", "mean_est_weighted_aadj")
    ]
).with_columns(
    [
        (pl.col(x)/pl.col("true_mean")).alias(f"rel{x}")
        for x in ("error_mean_est_weighted_adj2", "error_mean_est_weighted_adj", "error_mean_est_weighted", "error_mean_est_unweighted", "error_mean_est_weighted_aadj")
    ]
)

combined_summary.write_csv("combined_summary.csv")

# todo: join back to a table of query info here so we can use it in charts... maybe, or could use the actual mean
normalized_combined_summary = combined_summary.unpivot(index=["QID", "phase", "run", "sample_period", "sample_frequency"])
normalized_combined_summary.write_csv("ncs.csv")


# todo: note a plot split out by query id or mean duration actually hides some of the bias
# todo: although saying that those are suppose to represent different queries so I would want to split them in real life

# sns.boxplot(
#     normalized_combined_summary.filter(pl.col("variable").str.contains("relerror")),
#     # x="query_length_in_periods",
#     x="variable",
#     y="value",
#     native_scale=False,
#     whis=(5, 95),
#     fliersize=0,
#     color="#cfe2f3ff",
#     # hue="QID"
#     # linecolor=inner_lines,
#     # **hue_params,
# )

sns.boxplot(
    normalized_combined_summary.filter(pl.col("variable")=="relerror_mean_est_weighted_adj"),
    # x="query_length_in_periods",
    x="sample_frequency",
    y="value",
    native_scale=True,
    whis=(5, 95),
    fliersize=0,
    color="#cfe2f3ff",
    hue="QID"
    # linecolor=inner_lines,
    # **hue_params,
)

# print(normalized_combined_summary.group_by(['QID', 'variable']).agg([
#             pl.mean("value").alias("mean"),
#             pl.std("value").alias("std"),
#         ]).filter(pl.col('variable').is_in('true_mean', 'mean_est_unweighted').sort(by='QID'))
#
pt.show()

# todo: pivot in a different way so we can do a scatter of true mean vs estimated mean

# todo: something weird here that I can't quite get my head round, the relerror for unweighted always seems to be about +100% with exponential when queries are much shorter than the sampling period so, perhaps there is a simpler rule I should be using?
# todo: it also doesn't seem to correspond much to the mean query duration... but that's expected because it's the variance which should drive bias


# Because we know the true duration of the sampled queries as well as the true duration of all queries we can see if there's any obvious relationship between the two
# Haven't found one yet... they do seem kind of related by the variance, but not in a very obvious way
