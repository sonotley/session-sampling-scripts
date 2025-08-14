"""Script to generate vectors representing distributions of estimated latency and write them to pgVector and to numpy arrays in a pickle"""

from itertools import product
import pickle
import psycopg2 as pg
from psycopg2.extras import execute_values

import numpy as np

from query_latency_plots import generate_est_dist_curve_from_true_dist, generate_dist_curve

dists = []

for mean, z in product(
    np.arange(start=10, stop=1500, step=2), np.arange(start=0.1, stop=10, step=0.2)
):
    # limit the sd so we don't waste lots of space with extreme dists
    if 10 < mean * z < 3000:

        dist = generate_est_dist_curve_from_true_dist(
            generate_dist_curve("ln", mean=mean, spread=mean * z, size=4000), 1000
        )
        dists.append((mean, mean * z, dist))

# todo: set correct dist type ID for other dist types
dists_pgvector = (
    ((float(x[0]), float(x[0] * x[1]), 1, x[2].astype(np.float16).tolist()))
    for x in dists
)
print(f"Attempting to insert {len(dists)} histograms in Postgres")
conn = pg.connect("postgres://simon@localhost/simon")
curs = conn.cursor()
qry = "INSERT INTO dists_of_estimates(mean, sd, dist_type_id, dist) VALUES %s"
execute_values(curs, qry, dists_pgvector)
conn.commit()

with open("dists.pkl", "wb") as f:
    pickle.dump(dists, f)

# true_means = [x["mean"] for x in dists]
# est_means = [sum(x["dist"]*np.arange(start=0, stop=2000, step=1)) for x in dists]
# true_z = [x["sd"]/x["mean"] for x in dists]

# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.scatterplot(x=true_means, y=est_means, hue=true_z)
# plt.show()
