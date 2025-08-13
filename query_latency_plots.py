"""Not a script in its own right, contains functions to plot charts in query_latency.py"""

import math

import matplotlib.pyplot as pt
import numpy as np
import polars as pl
import seaborn as sns
from scipy.stats import gaussian_kde, lognorm, uniform, expon
from session_sampling_simulator import session_simulator as sim


def generate_dist_curve(
    dist: str, mean: float, spread: float, size: int, internal_size: int | None = None
) -> np.ndarray:
    if not internal_size:
        internal_size = size

    if dist.lower() in ("lognormal", "ln"):
        mu = math.log(mean**2 / math.sqrt(mean**2 + spread**2))
        sigma = np.sqrt(math.log(1 + (spread**2 / mean**2)))

        htd_base = lognorm(s=sigma, scale=np.exp(mu))
    elif dist.lower() in ("uniform", "u"):
        htd_base = uniform(loc=mean - spread, scale=2 * spread)
    elif dist.lower() in ("exponential", "ex"):
        htd_base = expon(scale=mean)
    return htd_base.pdf(range(internal_size)[:size])  # check for out by ones here


def generate_est_dist_curve_from_true_dist(true_dist: np.ndarray, sample_period: int):
    # todo: am I creating an issue by sizing the new dist to match the true one? Perhaps we should double it or something?
    dist = np.zeros(true_dist.size)
    # Here d represents a duration and x is the probability that the true duration is equal to i
    for d, x in enumerate(true_dist):
        # First we calculate the minimum and maximum number of times an execution of duration d could potentially be sampled
        # The max and min always differ by 1
        min_possible_samples = d // sample_period
        max_possible_samples = min_possible_samples + 1

        # The greatest value that 'a' can take whilst still fitting max_possible_samples into an execution of duration d
        a_cutoff = d % sample_period

        # The probability of an execution of duration d being sampled max_possible_samples times
        p_case_max = a_cutoff / sample_period

        # First the case where you get the maximum number of samples in
        c = int((max_possible_samples - 1) * sample_period)
        min_a = 0
        max_a = a_cutoff
        lower_limit = 2 * min_a + c
        upper_limit = 2 * max_a + c
        dist[lower_limit : upper_limit + 1] += (
            x * p_case_max / (upper_limit - lower_limit + 1)
        )

        # Then the case where you get one less than the max, unless that number would be zero
        # (in which case it means we don't sample the execution at all so we don't add any probabilty)
        if min_possible_samples > 0:
            p_case_min = 1 - p_case_max
            c = int(sample_period * (min_possible_samples - 1))
            min_a = (
                a_cutoff + 1
            )  # think it makes sense to have the +1 to prevent double counting
            max_a = sample_period
            lower_limit = 2 * min_a + c
            upper_limit = 2 * max_a + c
            dist[lower_limit : upper_limit + 1] += (
                p_case_min * x / (upper_limit - lower_limit + 1)
            )

    return dist / sum(dist)


def plot_distributions(
    execution_data: pl.DataFrame,
    sample_period,
    queries: list[sim.Query],
    bw=2,
    max_x=3000,
    query_ids=(1, 2, 3, 4, 5),
) -> None:
    fig, axes = pt.subplots(4, len(query_ids))
    cp = sns.color_palette("hls", 8)
    for i, qid in enumerate(query_ids):
        c = cp[i]
        # fixme: remove queries[i] here and use qid properly
        # fixme: implement an internal size distinct from max_x
        true_dist = generate_dist_curve(
            dist=queries[i].duration_distribution,
            spread=queries[i].duration_spread
            + 0.01,  # add 0.01 just to convince zero case to work
            mean=queries[i].mean_duration,
            size=max_x,
        )
        sns.histplot(
            execution_data.filter(pl.col("qid") == qid),
            x="true_duration",
            ax=axes[0, i],
            binwidth=bw,
            binrange=(0, max_x),
            color=c,
        )
        sns.lineplot(
            true_dist * len(execution_data.filter(pl.col("qid") == qid)) * bw,
            ax=axes[0, i],
            color="grey",
            linewidth=1,
        )
        sns.histplot(
            execution_data.filter(pl.col("qid_right") == qid),
            x="true_duration",
            ax=axes[1, i],
            binwidth=bw,
            binrange=(0, max_x),
            color=c,
        )
        sns.lineplot(
            true_dist * len(execution_data.filter(pl.col("qid_right") == qid)) * bw,
            ax=axes[1, i],
            color="grey",
            linestyle="dotted",
            linewidth=1,
        )
        exp_obs_dist = true_dist * [min(1, x / sample_period) for x in range(max_x)]
        exp_obs_dist /= sum(exp_obs_dist)
        sns.lineplot(
            exp_obs_dist * len(execution_data.filter(pl.col("qid_right") == qid)) * bw,
            ax=axes[1, i],
            color="grey",
            linewidth=1,
        )
        sns.histplot(
            execution_data.filter(pl.col("qid_right") == qid),
            x="estimate",
            ax=axes[2, i],
            binwidth=bw,
            binrange=(0, max_x),
            color=c,
        )
        sns.lineplot(
            generate_est_dist_curve_from_true_dist(
                true_dist,
                sample_period=sample_period,
            )
            * len(execution_data.filter(pl.col("qid_right") == qid))
            * bw,
            ax=axes[2, i],
            color="grey",
            linewidth=1,
        )
        sns.lineplot(
            generate_est_dist_curve_from_true_dist(
                true_dist,
                sample_period=sample_period,
            ),
            ax=axes[3, i],
            color=c,
        )
        raw_est = (
            execution_data.filter(pl.col("qid_right") == qid)
            .select("estimate")
            .to_numpy()
            .T
        )
        kde = gaussian_kde(np.concat((raw_est, -raw_est), axis=1), bw_method=0.02)
        sns.lineplot(
            2 * kde.pdf(range(max_x)),
            ax=axes[3, i],
            color="grey",
            linewidth=1,
        )

    pt.show()
