import numpy as np

from session_sampling_simulator import session_simulator as sim
import polars as pl
from milestone_timer import MilestoneTimer

mt = MilestoneTimer()
print(mt.add_milestone("start"))

DURATION = 3600000
SAMPLE_PERIOD = 1000

queries = sim.load_queries_from_file("queries.example.yml")

sess = sim.generate_session(queries=queries, window_duration=DURATION)

sess = sess // 10  # remove the wait ID for now

# true_regions = sim.find_contiguous_regions(sess)

st = sim.get_sample_times(session_duration=DURATION, sample_period=SAMPLE_PERIOD)

print(mt.add_milestone("generated session"))

sample = sess[st]

query_elapsed_times = zip(
    st[:-1],
    sample[:-1],
    sample[:-1] % 100,
    sample[:-1] // 100,
    st[:-1] - (sample[:-1] // 100) + 1,  # Added one to fudge div0 errors, need to think about this
    sample[1:] % 100,
)

headers = ["Timestamp", "Raw", "QID", "Start time", "Time since start", "Next QID"]

df = pl.DataFrame(query_elapsed_times, schema=headers)
df2 = df.filter(pl.col("QID") != 0)
df2.write_csv("results.csv")
df3 = (
    df2.group_by(by="Start time")
    .agg(
        [
            pl.min("Time since start").alias("a"),
            pl.min("Timestamp").alias("First"),
            pl.max("Timestamp").alias("Last"),
        ]
    )
    .with_columns((2 * pl.col("a") + pl.col("Last") - pl.col("First")).alias("Estimate"))
    .with_columns((pl.max_horizontal(SAMPLE_PERIOD / pl.col("Estimate"), 1)).alias("Weight"))
    .with_columns((pl.col("Estimate") * pl.col("Weight")).alias("weighted_est"))  # this is pointless as it just cancels out to the sample period
)
df3.write_csv("results2.csv")

print(df3.describe())

# something not right with my weighting or something... this comes out around 50% of the correct value
# the mean of a is remarkably close to the correct value... is that just a weird coincidence?

# when the queries have no variance, the mean(2est) value looks good... but the weighted one is wrong still
# It looks like my weighting is adding spurious variance where there is none and biasing the estimate
# I need some way to account for the variance that comes from the random observation... this is getting super complex

# todo: add grouping by QID here
print(df3.select(pl.sum("weighted_est")) / df3.select(pl.sum("Weight")))
