"""Functions for performing a similarity search given a histogram

The functions provided by this module are closures that set up the necessary in-memory structures
(a database connection, or a dictionary of known histograms) and return a function to perform
a search against them.

"""

import psycopg2 as pg
import numpy as np
import pickle


def get_db_search_function(dsn: str) -> callable:
    conn = pg.connect(dsn)
    curs = conn.cursor()

    def find_similar_in_db(dist):
        curs.execute(
            "SELECT mean, sd FROM dists_of_estimates ORDER BY dist <-> %s::halfvec LIMIT 1",
            (dist.astype(np.float16).tolist(),),
        )
        return curs.fetchone()

    return find_similar_in_db


def get_brute_force_function(fp: str) -> callable:
    with open(fp, "rb") as f:
        dists = pickle.load(f)

    def find_similar_in_pickle(dist: np.ndarray) -> tuple[float, float]:
        min_diff = 1
        for d in dists:
            diff = np.sum((d[2] - dist) ** 2)
            if diff < min_diff:
                min_diff = diff
                best = (d[0], d[1])
        return best

    return find_similar_in_pickle
