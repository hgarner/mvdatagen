import data_generation_mvnorm as dg
import numpy as np
import pandas as pd

# 20 samples, 12 vars
data_mv = dg.correlated_mvnorm(num_vars = 12, neg_cols = (0, 2), pos_cols = [(3, 9), (11, 12)], neg_range = (-1, -0.6), pos_range = (0.6, 1), low_range = (-0.3, 0.3), means = 3, sample_size = 20)

# convert some cols to lognormal
data_mv = dg.convert_lognorm(subset, cols = [(6, 8)])

# convert some cols to bimodal beta distribution
data_mv = dg.convert_norm_to_dist(subset, means = 3, sd = 1.0, cols = [(2, 4)], dist = 'beta', params = {'a': 0.5, 'b': 0.5})

