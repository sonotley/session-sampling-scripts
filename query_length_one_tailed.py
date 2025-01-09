import numpy as np
from matplotlib import pyplot as pt
import seaborn as sns

from session_sampling_simulator import session_simulator as sim
import polars as pl

DURATION = 3600000
SAMPLE_PERIOD = 1000

queries = sim.load_queries_from_file("queries.example.yml")

results = []

for run in range(1, 100):
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
        pl.mean("true_duration").alias("true_mean")
    )

    for phase in np.linspace(0, 0.99, 10):
        st = sim.get_sample_times(
            session_duration=DURATION, sample_period=SAMPLE_PERIOD, phase=phase
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

        non_zero_samples = pl.DataFrame(query_elapsed_times, schema=headers).filter(pl.col("QID") != 0).join(true_regions, on="start")        # df2.write_csv("results.csv")
        observed_queries = (
            non_zero_samples.group_by(["start", "QID", "end"])
            .agg(
                [
                    pl.min("Time since start").alias("a"),
                    pl.min("Timestamp").alias("First"),
                    pl.max("Timestamp").alias("Last"),
                ]
            )
            .with_columns((pl.col("Last") - pl.col("First")).alias("c"))
            .with_columns((pl.col("end") - pl.col("start") + 1).alias("true_duration"))
            .with_columns((2 * pl.col("a") + pl.col("c")).alias("estimate"))
            .with_columns(
                (pl.max_horizontal(SAMPLE_PERIOD / pl.col("estimate"), 1)).alias(
                    "weight"
                )
            )
            .with_columns((pl.col("estimate") * pl.col("weight")).alias("weighted_est"))
            .with_columns((pl.col("estimate").mean().over("QID")).alias("est_mean"))
            .with_columns((pl.col("estimate").var().over("QID")).alias("est_var"))
            .with_columns((pl.col("a").mean().over("QID")).alias("a_mean"))
            .with_columns(((16 / 12) * (pl.col("a_mean") ** 2)).alias("exp_var"))
            # This is all a bit of a random bodge, but it seems to be more or less correct
            .with_columns(
                (
                    pl.col("est_mean")
                    + (
                        np.sqrt(
                            1 - pl.col("exp_var") / pl.col("est_var")
                        )  # unsure if should be sqrt here - the correct answer seems to be between having and not having it
                        * (pl.col("estimate") - pl.col("est_mean"))
                    )
                ).alias("est_adj")
            )
            .with_columns(
                (pl.max_horizontal(SAMPLE_PERIOD / pl.col("est_adj"), 1)).alias(
                    "weight_adj"
                )
            )
            .with_columns(
                (pl.col("estimate") * pl.col("weight_adj")).alias("weighted_est_adj")
            )
        )
        # df3.write_csv("results2.csv")

        # print(df3.describe())

        summary_by_qid = (
            observed_queries.group_by(["QID"])
            .agg(
                [
                    pl.mean("true_duration").alias("true_duration_sampled_only"),
                    pl.mean("estimate").alias("mean_est_unweighted"),
                    (pl.sum("weighted_est") / pl.sum("weight")).alias(
                        "mean_est_weighted"
                    ),
                    (pl.sum("weighted_est_adj") / pl.sum("weight_adj")).alias(
                        "mean_est_weighted_adj"
                    ),
                ]
            )
            .join(true_summary, on="QID")
            .with_columns(phase=phase, run=run)
        )

        # df4.write_csv("summary.csv")
        results.append(summary_by_qid)

combined_summary = pl.concat(results).with_columns(
    [
        (pl.col(x) - pl.col("true_mean")).alias(f"error-{x}")
        for x in ("mean_est_weighted_adj", "mean_est_weighted", "mean_est_unweighted")
    ]
)

# combined_summary.write_csv("many.csv")

normalized_combined_summary = combined_summary.unpivot(index=["QID", "phase", "run"])
# df6.write_csv("unpivot.csv")


sns.boxplot(
    normalized_combined_summary,
    # x="query_length_in_periods",
    x="value",
    y="variable",
    native_scale=False,
    whis=(5, 95),
    fliersize=0,
    color="#cfe2f3ff",
    # linecolor=inner_lines,
    # **hue_params,
)

pt.show()

# Because we know the true duration of the sampled queries as well as the true duration of all queries we can see if there's any obvious relationship between the two