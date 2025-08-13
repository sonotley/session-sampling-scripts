"""Main script for my query latency adventures.

Contains lots of functions to run simulated sessions from my session_sampling_simulator library
and calculated the true and estimated query latencies.

"""

from collections.abc import Iterable
import itertools
import datetime
import os
from pathlib import Path

import numpy as np
import polars as pl

from session_sampling_simulator import session_simulator as sim
from milestone_timer import MilestoneTimer
from query_latency_plots import plot_distributions
from snignificant import round_to_sf, round_to_position
from vector_search import get_brute_force_function, get_db_search_function


mt = MilestoneTimer()


def make_query(
    id: int,
    mean_duration: int,
    duration_spread: int,
    session_proportion: float,
    duration_dist: str,
) -> sim.Query:
    return sim.Query(
        id=id,
        mean_duration=mean_duration,
        target_periodicity=int(mean_duration / session_proportion),
        wait_state_ratios={1: 1},
        duration_distribution=duration_dist,
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
        session_duration=len(session),
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


# todo: this still has some out by one issues possibly - think i've got them
# todo: instead of enumerating the true dist it would be better for it to come with an array of bin centres
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
        # (
        #     100
        #     * (pl.mean("estimate") - pl.col("true_duration"))
        #     / pl.col("true_duration")
        # )
        # .filter(pl.col("estimate").is_not_null())
        # .mean()
        # .alias("perc_per_execution_error_obs"),
        # (
        #     100
        #     * (pl.mean("estimate") - pl.col("true_duration"))
        #     / pl.col("true_duration")
        # )
        # .mean()
        # .alias("perc_per_execution_error"),
    ]
    if include_weighted:
        if not sample_period:
            raise ValueError(
                "sample_period must be specified to calculate weighted estimates"
            )
        r = execution_data.group_by(["qid", "phase"]).agg(
            core
            + [
                (
                    (
                        pl.col("estimate")
                        * pl.max_horizontal(sample_period / pl.col("estimate"), 1)
                    ).sum()
                    / (pl.max_horizontal(sample_period / pl.col("estimate"), 1)).sum()
                ).alias("est_mean_weighted"),
            ]
        )
    else:
        r = execution_data.group_by(["qid", "phase"]).agg(core)

    return r


def flatten(l: Iterable[Iterable]):
    return [item for sublist in l for item in sublist]


def summarise_many_runs(summary_data: pl.DataFrame) -> pl.DataFrame:
    aggs = flatten(
        [
            (
                pl.mean(x),
                pl.std(x).alias(f"{x}_sd"),
                (
                    (100 * (pl.col(x) - pl.col("true_mean")) / pl.col("true_mean"))
                    .mean()
                    .alias(f"perc_error_{x}")
                ),
                (
                    (100 * (pl.col(x) - pl.col("true_mean")) / pl.col("true_mean"))
                    .std()
                    .alias(f"perc_error_{x}_sd")
                ),
            )
            for x in summary_data.columns
            if "mean" in x or "error" in x
        ]
    )
    return summary_data.group_by("qid").agg(aggs).sort("qid")


def generate_sampled_session(
    queries: list[sim.Query],
    duration: int,
    sample_period: int,
    phase: float | Iterable[float] = 0,
) -> pl.DataFrame:
    sess = sim.generate_session(queries=queries, window_duration=duration)

    sess = sess // 10  # remove the wait ID for now

    all_executions = find_all_query_executions(sess)

    phases = phase if isinstance(phase, Iterable) else [phase]

    chunks = []

    for p in phases:
        samples = extract_sample_data(
            session=sess, sample_period=sample_period, phase=p
        )

        sampled_executions = roll_up_samples_to_executions(samples)

        chunks.append(
            all_executions.join(
                sampled_executions, on="start", how="left"
            ).with_columns(phase=p)
        )

    return pl.concat(chunks)


def pretty_print_mean_and_sd(mean, sd) -> str:
    rounded_sd = round_to_sf(sd)
    rounded_mean = round_to_position(mean, rounded_sd.exponent)
    return f"{rounded_mean.as_str} ± {rounded_sd.as_str}"


def pretty_print_summary(summary: pl.DataFrame):
    formatted_rows = []
    for row in summary.iter_rows(named=True):
        formatted_row = {"qid": row["qid"]}
        for col in [x for x in row if "_sd" in x and x != "perc_error_true_mean_sd"]:
            formatted_row[col[:-3]] = pretty_print_mean_and_sd(row[col[:-3]], row[col])
        formatted_rows.append(formatted_row)

    a = pl.DataFrame(formatted_rows)

    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_cols=-1,
    ):
        print(a)


def executions_to_hist(execution_data: pl.DataFrame, bins: int) -> np.ndarray:
    return np.histogram(
        execution_data.select("estimate").to_numpy(),
        bins=range(bins + 1),
        density=True,
    )[0]


def create_datetime_directory(root: Path, dt=None):
    """
    Creates a new directory in the current working directory,
    named after the current datetime in YYYYMMDDHHMMSS format.
    """
    now = dt or datetime.datetime.now()
    dir_name = now.strftime("%Y%m%d%H%M%S")

    try:
        os.makedirs(root / dir_name)
    except OSError as e:
        print(f"Error creating directory '{dir_name}': {e}")

    return root / dir_name


def perform_runs(
    queries: list[sim.Query],
    duration: int | None = None,
    sample_period: int | None = None,
    number_of_runs: int | None = None,
    number_of_phases: int | None = None,
    save_run_data: bool = False,
    read_run_data_from: Path | None = None,
    vector_search_function=None,
):
    if not read_run_data_from and not (
        queries and duration and sample_period and number_of_runs and number_of_phases
    ):
        raise ValueError(
            "When not reading from a directory, you must specify all parameters."
        )

    if save_run_data:
        save_dir = create_datetime_directory(Path() / "saved")

    phases = np.linspace(start=0, stop=1, endpoint=False, num=number_of_phases)

    run_data = []
    run_summaries = []

    print(mt.add_milestone("Queries ready"))

    for run in range(number_of_runs):
        if read_run_data_from:
            run_executions_augmented = pl.read_parquet(read_run_data_from / f"run{run}")
        else:
            run_executions_augmented = generate_sampled_session(
                queries=queries,
                duration=duration,
                sample_period=sample_period,
                phase=phases,
            )

            if save_run_data:
                run_executions_augmented.write_parquet(save_dir / f"run{run}")

        run_summary = summarise_run(
            run_executions_augmented, include_weighted=True, sample_period=sample_period
        )

        print(mt.add_milestone("Session simulation complete"))

        if vector_search_function:
            qids = run_summary.select(pl.col("qid").unique()).to_series().to_list()
            # todo: using the KDE for the search would probably be far better
            vector_ests = []
            for q, p in itertools.product(qids, phases):
                observed_dist = executions_to_hist(
                    run_executions_augmented.filter(
                        (pl.col("qid_right") == q) & (pl.col("phase") == p)
                    ),
                    bins=4000,
                )
                vector_ests.append((q, p, vector_search_function(observed_dist)[0]))

            vector_df = pl.DataFrame(
                vector_ests, schema=["qid", "phase", "est_mean_vector"], orient="row"
            )

            print(mt.add_milestone("Vector estimation complete"))

            #  todo: run summary doesn't incluse phase, need to add option to preserve it
            run_summaries.append(
                run_summary.with_columns(run=run).join(vector_df, on=["qid", "phase"])
            )

        else:
            #  todo: run summary doesn't incluse phase, need to add option to preserve it
            run_summaries.append(run_summary.with_columns(run=run))

        run_data.append(run_executions_augmented.with_columns(run=run))

        print(mt.add_milestone("Summarisation complete"))

    print(mt.add_milestone("All runs complete"))

    all_executions_augmented = pl.concat(run_data)
    runs = pl.concat(run_summaries)

    print(mt.add_milestone("Data concatenated"))

    return all_executions_augmented, runs


if __name__ == "__main__":
    sample_period = 1000

    queries = [
        make_query(
            id=i,
            mean_duration=50 + 250 * (i - 1),
            duration_spread=300,
            session_proportion=0.05,
            duration_dist="lognormal",
        )
        for i in range(1, 6)
    ]

    # vector_searcher = get_brute_force_function("dists.pkl")
    vector_searcher = get_db_search_function("postgres://simon@localhost/simon")

    all_executions_augmented, runs = perform_runs(
        queries=queries,
        duration=3600000,
        sample_period=1000,
        number_of_runs=1,
        number_of_phases=5,
        # vector_search_function=vector_searcher,
    )

    # all_executions_augmented.write_database(table_name='execs', if_table_exists='replace', connection="postgresql://postgres:postgres@localhost/simon")

    # print(mt.add_milestone("Data written to Postgres"))

    plot_distributions(
        all_executions_augmented,
        sample_period,
        queries=queries,
        bw=10,
        query_ids=(1, 2, 3, 4, 5),
    )

    pretty_print_summary(summarise_many_runs(runs))

    # print(summarise_run(all_executions_augmented, include_weighted=True, sample_period=sample_period))
