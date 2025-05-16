from itertools import product
import pickle

import numpy as np

from query_latency import generate_est_dist_from_true_dist, generate_lognorm

dists = []

for mean, z in product(
    np.arange(start=5, stop=1500, step=5), np.arange(start=0.1, stop=2, step=0.1)
):
    dist = generate_est_dist_from_true_dist(
        generate_lognorm(mean, mean * z, 2000), 1000
    )
    dists.append({"mean": mean, "sd": mean * z, "dist": dist})

# with open("dists.pkl", "wb") as f:
#     pickle.dump(dists, f)

true_means = [x["mean"] for x in dists]
est_means = [sum(x["dist"]*np.arange(start=0, stop=2000, step=1)) for x in dists]
true_z = [x["sd"]/x["mean"] for x in dists]

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=true_means, y=est_means, hue=true_z)
plt.show()