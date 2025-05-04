import math
import pickle

import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as pt
from scipy.stats import lognorm, gaussian_kde

from session_sampling_simulator import session_simulator as sim


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


def find_all_query_executions(session: np.ndarray) -> pl.DataFrame:
    return (
        pl.DataFrame(
            sim.find_contiguous_regions(session),
            orient="row",
            schema=["start", "end", "literal"],
        )
        .with_columns(qid=pl.col("literal") % 100)
        .with_columns((pl.col("end") - pl.col("start") + 1).alias("true_duration"))
        .filter(pl.col("qid") != 0)
    )


def extract_sample_data(
    session: np.ndarray, sample_period: int, phase: float
) -> pl.DataFrame:
    st = sim.get_sample_times(
        session_duration=DURATION,
        sample_period=sample_period,
        phase=phase,
        strategy=sim.SamplingStrategy.UNIFORM,
    )

    sample = session[st]

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
        "timestamp",
        "raw",
        "qid",
        "start",
        "time since start",
        "next qid",
    ]

    # Select only the observations where the QID was non-zero
    # i.e. where we've observed a query execution
    return pl.DataFrame(sample_data, schema=headers).filter(pl.col("qid") != 0)


def roll_up_samples_to_executions(sample_data: pl.DataFrame) -> pl.DataFrame:
    return (
        sample_data.group_by(["start", "qid"])
        .agg(
            [
                pl.min("time since start").alias("a"),
                pl.min("timestamp").alias("first"),
                pl.max("timestamp").alias("last"),
            ]
        )
        .with_columns((pl.col("last") - pl.col("first")).alias("c"))
        .with_columns((2 * pl.col("a") + pl.col("c")).alias("estimate"))
    )


def generate_lognorm(mean: float, sd: float, length: int) -> np.ndarray:
    mu = math.log(mean**2 / math.sqrt(mean**2 + sd**2))
    sigma = np.sqrt(math.log(1 + (sd**2 / mean**2)))

    htd_base = lognorm(s=sigma, scale=np.exp(mu))
    return htd_base.pdf(range(length))


# todo: this still has some out by one issues
# todo: this is wrong somewhere, it's not correctly dealing with the scenario of getting less than the max number
def generate_obs_dist_from_true_dist(true_dist: np.ndarray, sample_period: int):
    dist = np.zeros(true_dist.size)
    for i, x in enumerate(true_dist):
        # i = j + 1

        min_possible_samples = i // sample_period
        max_possible_samples = min_possible_samples + 1
        a_cutoff = i % sample_period
        p_case_max = a_cutoff / sample_period

        # First the case where you get the maximum number of samples in
        # if max_possible_samples > 1:
        c = int((max_possible_samples - 1) * sample_period)
        min_a = 0
        max_a = a_cutoff
        lower_limit = 2 * min_a + c
        upper_limit = 2 * max_a + c
        dist[lower_limit:upper_limit] += x * p_case_max / (upper_limit - lower_limit)

        # Then the case where you get one less than the max
        if min_possible_samples > 0:
            p_case_min = 1 - p_case_max
            c = int(sample_period * (min_possible_samples - 1))
            min_a = a_cutoff  # todo: maybe +1 to be picky?
            max_a = sample_period
            lower_limit = 2 * min_a + c
            upper_limit = 2 * max_a + c
            dist[lower_limit:upper_limit] += (
                p_case_min * x / (upper_limit - lower_limit)
            )

    return dist / sum(dist)


def plot_distributions(
    execution_data: pl.DataFrame, sample_period, bw=2, max_x=1500
) -> None:
    fig, axes = pt.subplots(4, 5)
    cp = sns.color_palette("hls", 8)
    for i in range(1, 6):
        c = cp[i - 1]
        true_dist = generate_lognorm(
            sd=queries[i - 1].duration_spread
            + 0.01,  # add 0.01 just to convince zero case to work
            mean=queries[i - 1].mean_duration,
            length=max_x,
        )
        sns.histplot(
            execution_data.filter(pl.col("qid") == i),
            x="true_duration",
            ax=axes[0, i - 1],
            binwidth=bw,
            binrange=(0, max_x),
            color=c,
        )
        sns.lineplot(
            true_dist * len(execution_data.filter(pl.col("qid") == i)) * bw,
            ax=axes[0, i - 1],
            color="grey",
            linewidth=1,
        )
        sns.histplot(
            execution_data.filter(pl.col("qid_right") == i),
            x="true_duration",
            ax=axes[1, i - 1],
            binwidth=bw,
            binrange=(0, max_x),
            color=c,
        )
        sns.lineplot(
            true_dist * len(execution_data.filter(pl.col("qid_right") == i)) * bw,
            ax=axes[1, i - 1],
            color="grey",
            linestyle="dotted",
            linewidth=1,
        )
        exp_obs_dist = true_dist * [min(1, x / sample_period) for x in range(max_x)]
        exp_obs_dist /= sum(exp_obs_dist)
        sns.lineplot(
            exp_obs_dist * len(execution_data.filter(pl.col("qid_right") == i)) * bw,
            ax=axes[1, i - 1],
            color="grey",
            linewidth=1,
        )
        sns.histplot(
            execution_data.filter(pl.col("qid_right") == i),
            x="estimate",
            ax=axes[2, i - 1],
            binwidth=bw,
            binrange=(0, max_x),
            color=c,
        )
        sns.lineplot(
            generate_obs_dist_from_true_dist(
                true_dist,
                sample_period=sample_period,
            )
            * len(execution_data.filter(pl.col("qid_right") == i))
            * bw,
            ax=axes[2, i - 1],
            color="grey",
            linewidth=1,
        )
        sns.lineplot(
            generate_obs_dist_from_true_dist(
                true_dist,
                sample_period=sample_period,
            ),
            ax=axes[3, i - 1],
            color=c,
        )
        raw_est = (
            execution_data.filter(pl.col("qid_right") == i)
            .select("estimate")
            .to_numpy()
            .T
        )
        kde = gaussian_kde(np.concat((raw_est, -raw_est), axis=1), bw_method=0.02)
        print(kde.factor)
        sns.lineplot(
            2 * kde.pdf(range(max_x)),
            ax=axes[3, i - 1],
            color="grey",
            linewidth=1,
        )

    pt.show()


def test_dist(execution_data: pl.DataFrame, qid: int):
    min_diff = 1
    with open("dists.pkl", "rb") as f:
        dists = pickle.load(f)
    # this is actually a vector similarity search... intrigiung
    observed = np.histogram(
        execution_data.filter(pl.col("qid_right") == qid).select("estimate").to_numpy(),
        bins=range(0, 2001),
        density=True,
    )[0]
    for d in dists:
        diff = np.sum((d["dist"] - observed) ** 2)
        # print(d["mean"], d["sd"], diff)
        if diff < min_diff:
            min_diff = diff
            best = (d["mean"], d["sd"])
    print(best)


def summarise_run(
    execution_data: pl.DataFrame,
    include_weighted: bool = False,
    sample_period: int = None,
):
    core = [
        pl.mean("true_duration").alias("true_mean"),
        (pl.col("true_duration").filter(pl.col("estimate").is_not_null()))
        .mean()
        .alias("true_mean_obs"),
        pl.mean("estimate").alias("est_mean"),
        # pl.std("estimate").alias("est_sd"),
    ]
    if include_weighted:
        if not sample_period:
            raise ValueError(
                "sample_period must be specified to calculate weighted estimates"
            )
        r = execution_data.group_by("qid").agg(
            core
            + [
                (
                    (
                        (
                            (
                                pl.col("estimate")
                                * pl.max_horizontal(
                                    sample_period / pl.col("estimate"), 1
                                )
                            ).sum()
                        )
                        / (
                            (
                                pl.max_horizontal(sample_period / pl.col("estimate"), 1)
                            ).sum()
                        )
                    ).alias("est_mean_weighted")
                )
            ]
        )
    else:
        r = execution_data.group_by("qid").agg(core)

    return r

def flatten(l):
    return [item for sublist in l for item in sublist]

def summarise_many_runs(summary_data: pl.DataFrame) -> pl.DataFrame:
    aggs = flatten([(pl.mean(x), pl.std(x).alias(f"{x}_sd")) for x in summary_data.columns if "mean" in x])
    return (
        summary_data.group_by("qid")
        .agg(aggs)
        .sort("qid")
    )


# todo: support multiple phases and sample_periods
def generate_sampled_session(
    queries: list[sim.Query], duration: int, sample_period: int
) -> pl.DataFrame:
    sess = sim.generate_session(queries=queries, window_duration=duration)

    sess = sess // 10  # remove the wait ID for now

    all_executions = find_all_query_executions(sess)

    samples = extract_sample_data(session=sess, sample_period=sample_period, phase=0)

    sampled_executions = roll_up_samples_to_executions(samples)

    return all_executions.join(sampled_executions, on="start", how="left")


def pretty_print_mean_and_sd(mean, sd) -> str:
    return f"{int(mean)} ± {int(sd)}"

def pretty_print_summary(summary: pl.DataFrame):
    formatted_rows = []
    for row in summary.iter_rows(named=True):
        formatted_row = {"qid": row["qid"]}
        for col in [x for x in row if "_sd" in x]:
            formatted_row[col[:-3]] = pretty_print_mean_and_sd(row[col[:-3]], row[col])
        formatted_rows.append(formatted_row)

    a = pl.DataFrame(formatted_rows)

    with pl.Config(
    tbl_formatting="MARKDOWN",
    tbl_hide_column_data_types=True,
    tbl_hide_dataframe_shape=True,):
        print(a)


if __name__ == "__main__":
    DURATION = 3600000
    sample_period = 1000

    queries = [
        make_query(
            id=i, mean_duration=40 * i, duration_spread=40 * i, session_proportion=0.1
        )
        for i in range(1, 6)
    ]

    run_data = []
    run_summaries = []

    for run in range(5):
        run_executions_augmented = generate_sampled_session(
            queries=queries, duration=DURATION, sample_period=sample_period
        )
        run_summary = summarise_run(run_executions_augmented)
        run_data.append(run_executions_augmented.with_columns(run=run))
        run_summaries.append(run_summary.with_columns(run=run))

    all_executions_augmented = pl.concat(run_data)
    runs = pl.concat(run_summaries)

    # all_executions_augmented.write_database(table_name='execs', connection="postgresql://postgres:postgres@localhost/simon")

    # plot_distributions(all_executions_augmented, sample_period)

    pretty_print_summary(summarise_many_runs(runs))

    # print(summarise_run(all_executions_augmented, include_weighted=True, sample_period=sample_period))

    # for i in range(1, 6):
    #     test_dist(all_executions_augmented, i)
