import data_generation_mvnorm as dg
import numpy as np
import pandas as pd

# 20 samples, 12 vars
# note that correlated_mvnorm returns a tuple of the correlation matrix and the data
data_mv = dg.correlated_mvnorm(num_vars = 12, neg_cols = (0, 2), pos_cols = [(3, 9), (11, 12)], neg_range = (-1, -0.6), pos_range = (0.6, 1), low_range = (-0.3, 0.3), means = 3, sample_size = 20)

# convert some cols to lognormal
data_mv_lognorm = dg.convert_lognorm(data_mv[1], cols = [(6, 8)])

# convert some cols to bimodal beta distribution
data_mv_beta = dg.convert_norm_to_dist(data_mv[1], means = 3.0, sd = 1.0, cols = [(2, 4)], dist = 'beta', params = {'a': 0.5, 'b': 0.5})

# more options using the CorrMatrix object and gen_correlated_mvnorm
num_vars = 12
# generate a CorrMatrix object
# (above just uses the gen_corr_matrix function)
corr = dg.CorrMatrix()

corr.range_settings['high_pos'] = (1, 20)
corr.range_settings['high_neg'] = (-20, -1)
corr.create_w(num_vars = num_vars)
corr.set_cols_to_range(range_id = 'high_pos', cols = [(0, 3)])
corr.set_cols_to_range(range_id = 'high_neg', cols = [(3, 6)])
corr.set_cols_to_range(range_id = 'mid_pos', cols = [(6, 9)])
corr.set_cols_to_range(range_id = 'mid_neg', cols = [(9, 12)])
# generate the correlation matrix from this
corr.gen_corr_matrix()
# generated matrix
corr.corr

# 200 samples
# create a set of random means between 2.5 and 3.5
means_set = np.random.uniform(2.5, 3.5, num_vars)

# generate the data
# note that correlated_mvnorm returns a tuple of the correlation matrix and the data
data = dg.gen_correlated_mvnorm(corr = corr.corr, means = means_set, sample_size = 200)

