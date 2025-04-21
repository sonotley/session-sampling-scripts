from itertools import product
import pickle

from query_latency import generate_obs_dist_from_true_dist, generate_lognorm

dists = []

for mean, z in product(range(1,10000), range(1,4)):
    dist = generate_obs_dist_from_true_dist(generate_lognorm(mean, mean*z, 2000), 1000)
    dists.append({"mean": mean, "sd": mean*z, "dist": dist})


with open("dists.pkl", "wb") as f:
    pickle.dump(dists, f)
