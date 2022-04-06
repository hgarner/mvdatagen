import numpy as np
import os, sys

from scipy import stats

# numpy random
# see https://numpy.org/doc/stable/reference/random/index.html
from numpy import random

####
# todo
#
# break this down into parts to return the matrix w as need this to adjust
# for second matrix
####
class CorrMatrix:
  ###
  # method from http://lisaling.site/2016/03/03/Simulating-data-following-a-given-covariance-structure/
  ###

  def __init__(self):
    self.corr = None
    self.cols = None
    self.neg_range = [(-20, -10)]
    self.pos_range = [(10, 20)]
    self.low_range = [(-0.1, 0.1)]
    self.num_vars = None
    self.norm_diag = 10.0
    self.range_settings = {
      'low': (-0.1, 0.1),
      'mid_pos': (0.3, 1.0),
      'mid_neg': (-1.0, -0.3),
      'high_pos': (1, 20),
      'high_neg': (-20, -1)
    }
    self._w = None

  @property
  def w(self):
    return self._w

  @w.setter
  def w(self, value):
    self._w = value
    self.cols = [None for i in range(0, self._w.shape[1])]
    self.set_cols_to_range(range_id = 'low', cols = [(0, self._w.shape[1])])

  @w.deleter
  def w(self):
    del self._w
    
  # create the columnwise matrix w
  # t(w) multipled by w forms the basis for the correlation matrix
  def create_w(self, num_vars: int) -> np.ndarray:
    # calling default_rng() gets new instance of generator
    # then call it's methods to obtain diff dists
    rng = np.random.default_rng()

    if num_vars < 1:
      raise ValueError('CorrMatrix.create_w: num_vars must be greater than 1')

    self.num_vars = num_vars

    # start with n*n matrix populated with low (close to 0) uniform random values
    w = np.random.uniform(low=self.range_settings['low'][0], high=self.range_settings['low'][1], size=(num_vars, num_vars))

    self.w = w

    return self.w

  # save the correlation matrix to filepath
  # return string filepath on success
  def save_corr(self, filepath: str) -> str:
    if not os.path.exists(os.path.dirname(filepath)):
      raise ValueError(f'CorrMatrix.save_corr: directory {os.path.dirname(filepath)} does not exist')
    
    with open(filepath, 'w') as corr_file:
      # should check the printoptions here as precision could be issue
      np.savetxt(corr_file, self.corr, delimiter = ',')

    return filepath

  # wrapper to set cols to ranges from self.range_settings
  # @param range_id a specified string range from 
  # self.range_settings
  # @param cols list of tuples for slices of column indices
  # then call set_columns
  def set_cols_to_range(self, range_id: str, cols: list) -> np.ndarray:
    # example cols:
    #  [
    #    (0,4), # slice so cols 0, 1, 2, 3
    #    (7,10) # slice so cols 7, 8, 9
    #  ]

    if range_id not in self.range_settings.keys():
      raise KeyError(f'CorrMatrix.set_cols_to_ranges: range_id {range_id} not in ranges lookup')

    self.set_columns(cols = cols, ranges = [self.range_settings[range_id] for col in cols])

    return self
    
  # set the specified columns of w to be values in specified ranges
  # cols list of tuples start and stop (exc) of cols to set to range
  # ranges list of tuples of ranges (low, high)
  def set_columns(self, cols: list, ranges: list) -> np.ndarray:

    if self.w is None:
      raise ValueError('CorrMatrix.set_columns: must call create_w before setting columns')

    num_vars = self.num_vars

    # if neg_ or pos_cols is single tuple, stick it in a list
    if isinstance(cols, tuple):
      cols = [cols]

    # as above with ranges
    if isinstance(ranges, tuple):
      ranges = [ranges]

    # fill out ranges to correct length
    # use low range from self.range_settings
    if len(ranges) < len(cols):
      ranges += [self.range_settings['low']  for i in range(1, len(neg_cols) - len(neg_range) + 1)]

    # cols to change
    # slice of cols (start, stop)
    for col_index, col in enumerate(cols):
      col = slice(*col)
      self.cols[col] = [
        {'range': ranges[col_index], 'isset': False}
        for c in self.cols[col]
      ]

    self.generate_cols()

    return self.w

  # generate values for columns of w using this.cols to set range
  # ignore if isset is True
  def generate_cols(self):
    for col_index, col in enumerate(self.cols):
      if not col['isset']:
        self.w[..., col_index] = np.random.uniform(low=col['range'][0], high=col['range'][1], size=(self.w.shape[1]))
        col['isset'] = True

    return self

  # generate a correlation matrix with the specified properties
  def gen_corr_matrix(self) -> np.ndarray:
    if not(isinstance(self.w, np.ndarray)):
      raise TypeError('CorrMatrix.gen_corr_matrx: w must be an instance of numpy.ndarray')
    
    w = self.w

    # to make +ve definite, multiply w transpose by w then 
    # add ones to diagonal (ensures diagonal is +ve)
    b = np.matmul(np.transpose(w), w) + np.diagflat(self.norm_diag * np.ones(self.num_vars))

    # normalise to 0 <= Xij <= 1
    corr = np.matmul(
      np.matmul(
        np.diagflat(1 / np.sqrt(np.diag(b))), 
        b
      ),
      np.diagflat(1 / np.sqrt(np.diag(b)))
    )

    self.corr = corr

    return corr


# generate a correlation matrix with the specified properties
# num_vars int number of vars to generate
# neg_cols list of tuples start and stop (exc) of cols to make negative corr
# pos_cols list of tuples start and stop (exc) of cols to make postive corr
# neg_range list of tuples low and high vals for neg_cols
# pos_range list of tuples low and high vals for pos_cols
# low_range tuple of low and high vals for default matrix vals
def gen_corr_matrix(num_vars: int, neg_cols: list, pos_cols: list, neg_range: list, pos_range: list, low_range: tuple) -> np.ndarray:

  ###
  # method from http://lisaling.site/2016/03/03/Simulating-data-following-a-given-covariance-structure/
  ###
  
  # calling default_rng() gets new instance of generator
  # then call it's methods to obtain diff dists
  rng = np.random.default_rng()

  # if neg_ or pos_cols is single tuple, stick it in a list
  if isinstance(neg_cols, tuple):
    neg_cols = [neg_cols]
  if isinstance(pos_cols, tuple):
    pos_cols = [pos_cols]

  # as above with ranges
  if isinstance(neg_range, tuple):
    neg_range = [neg_range]
  if isinstance(pos_range, tuple):
    pos_range = [pos_range]

  # fill out ranges to correct length
  # use last entry if not enough
  if len(neg_range) < len(neg_cols):
    neg_range += [neg_range[-1] for i in range(1, len(neg_cols) - len(neg_range) + 1)]

  if len(pos_range) < len(pos_cols):
    pos_range += [pos_range[-1] for i in range(1, len(pos_cols) - len(pos_range) + 1)]

  # start with n*n matrix populated with positive low uniform random values
  w = np.random.uniform(low=low_range[0], high=low_range[1], size=(num_vars, num_vars))

  # cols to make high -ve (towards -1)
  # slice of cols (start, stop)
  for (neg_index, neg_col) in enumerate(neg_cols):
    neg_col = slice(*neg_col)
    w[..., neg_col] = np.random.uniform(low=neg_range[neg_index][0], high=neg_range[neg_index][1], size=(num_vars, (neg_col.stop - neg_col.start)))

  # cols to make high +ve (towards 1)
  for (pos_index, pos_col) in enumerate(pos_cols):
    pos_col = slice(*pos_col)
    w[..., pos_col] = np.random.uniform(low=pos_range[pos_index][0], high=pos_range[pos_index][1], size=(num_vars, (pos_col.stop - pos_col.start)))

  # to make +ve definite, multiply w transpose by w then 
  # add ones to diagonal (ensures diagonal is +ve)
  b = np.matmul(np.transpose(w), w) + np.diagflat(np.ones(num_vars))

  # normalise to 0 <= Xij <= 1
  corr = np.matmul(
    np.matmul(
      np.diagflat(1 / np.sqrt(np.diag(b))), 
      b
    ),
    np.diagflat(1 / np.sqrt(np.diag(b)))
  )

  return corr

# generate num_vars of correlated data from a gaussian dist
# corr ndarray +ve definite correlation matrix
# means list means for data generated
# sample_size int sample size to be generated
# return ndarray shape num_vars x sample_size (i.e. vars are cols)
def gen_correlated_mvnorm(corr: np.ndarray, means: list, sample_size: int) -> np.ndarray:
  # calling default_rng() gets new instance of generator
  # then call it's methods to obtain diff dists
  rng = random.default_rng()

  # sort out means if not a list or doesn't have enough values
  if not isinstance(means, list) and not isinstance(means, np.ndarray):
    means = [means]

  if len(means) < corr.shape[0]:
    while len(means) < corr.shape[0]:
      means.append(means[-1])

  # get some multvariate normal data
  num_vars = len(corr)
  data = np.zeros((sample_size, num_vars))
  for i in range(0, sample_size):
    data[i] = rng.multivariate_normal(mean = means, cov = corr)

  return data

# convenience wrapper for gen_corr_matrix and gen_correlated_mvnorm
# generate num_vars of correlated data from a gaussian dist
# num_vars int number of vars to generate
# neg_cols list of tuples start and stop (exc) of cols to make negative corr
# pos_cols list of tuples start and stop (exc) of cols to make postive corr
# neg_range list of tuples low and high vals for neg_cols
# pos_range list of tuples low and high vals for pos_cols
# low_range tuple of low and high vals for default matrix vals
# mean list means for data generated. if float or only one entry, expand
# to fill all vars
# sample_size int sample size to be generated
# return tuple of correlation matrix (ndarray) and data (ndarray shape num_vars x sample_size (i.e. vars are cols))
def correlated_mvnorm(num_vars: int, neg_cols: list, pos_cols: list, neg_range: list, pos_range: list, low_range: list, means: list, sample_size: int) -> np.ndarray:

  corr = gen_corr_matrix(
    num_vars = num_vars,
    neg_cols = neg_cols,
    pos_cols = pos_cols,
    neg_range = neg_range,
    pos_range = pos_range,
    low_range = low_range
  )

  if not isinstance(means, list):
    means = [means]

  if len(means) < num_vars:
    while len(means) < num_vars:
      means.append(means[-1])

  data = gen_correlated_mvnorm(corr = corr, means = means, sample_size = sample_size)

  return (corr, data)

# convert specified columns of gaussian dataset to provided dist
# dataset numpy.ndarray dataset with columns as vars
# means list of floats specifying means of dataset
# sd standard deviation of cols # FIX THIS AS NEEDS TO BE VECTOR
# cols list of tuples start and stop (exc) of cols to convert
# dist string dist to apply (from scipy.stats)
# params dict params for dist
# return numpy.ndarray dataset
def convert_norm_to_dist(dataset: np.ndarray, means: np.ndarray, sd: float, cols: list, dist: str, params: dict) -> np.ndarray:
  if not isinstance(cols, list):
    if isinstance(cols, tuple):
      cols = [cols]
    else:
      raise TypeError('convert_cols_to_dist: cols must be a list of tuples, or s single tuple specifying cols to convert')

  # if means is a list, check it's the right length
  try:
    if len(means) < dataset.shape[1]:
      raise ValueError('convert_norm_to_dist: length of means is less than cols in dataset')
  except TypeError:
    # if the means is a float, put into a list of correct length
    if isinstance(means, float):
      means = np.array([means for i in range(0, dataset.shape[1])])
    else:
      raise TypeError('convert_norm_to_dist: means must be a single float or an ndarray/list of same length as dataset cols')

  for colset in cols:
    colset = slice(*colset)
    # convert the slice to a uniform dist by applying the normal cdf
    dataset[..., colset] = stats.norm.cdf(dataset[..., colset], means[colset], sd)

    # get the distribution's implementation from the stats module
    try:
      dist_funcs = getattr(stats, dist)
    except AttributeError:
      raise ValueError(f'convert_norm_to_dist: stats module has no methods for dist {dist}')

   # apply the inverse cdf to transform to the specified dist with params
    try:
      dataset[..., colset] = dist_funcs.ppf(dataset[..., colset], **params)
    except TypeError as te:
      raise ValueError(f'convert_norm_to_dist: incorrect params for dist {dist} ({te})')

  return dataset

# wrapper to call convert_norm_to_dist negative binomial distribution
# using mean and size (dispersion) params rather than count and prob
# for useful ranges of mean/size, see supplementary in 
# https://doi.org/10.1093/bioinformatics/btv165
def convert_nbinom_mean_size(mean, size, *args, **kwargs):
  # convert the mean and size to n and p
  # mean = n(1-p)/p
  # p = size/(size+mean)
  p = size / (size + mean)
  n = (p * mean) / (1 - p)
  return convert_norm_to_dist(*args, **kwargs, dist = 'nbinom', params = {'n': n, 'p': p})

# convert specified columns of gaussian dataset to lognormal dist
# dataset numpy.ndarray dataset with columns as vars
# cols list of tuples start and stop (exc) of cols to make lognormal
# could be wrapper for convert_norm_to_dist or just apply e^x
def convert_lognorm(dataset: np.ndarray, cols: list):
  if not isinstance(cols, list):
    if isinstance(cols, tuple):
      cols = [cols]
    else:
      raise TypeError('convert_lognorm: cols must be a list of tuples, or s single tuple specifying cols to convert')

  for colset in cols:
    colset = slice(*colset)
    # convert the slice to a lognorm 
    dataset[..., colset] = np.exp(dataset[..., colset])

  return dataset

def normalise_col(col: np.ndarray, min_to_zero = True) -> np.ndarray:
  if min_to_zero:
    col_min = col.min()
  else:
    col_min = 0
  col_max = col.max()
  return (col - col_min) / (col_max - col_min)

# normalise the columns of dataset to the range norm_range (tuple)
# return ndarray dataset
def normalise_cols(dataset: np.ndarray, norm_range = (0, 1), min_to_zero = True) -> np.ndarray:
  # not very efficient but it'll do
  if min_to_zero:
    norm_func = lambda col: normalise_col(col, min_to_zero = True)
  else:
    norm_func = lambda col: normalise_col(col, min_to_zero = False)

  return np.apply_along_axis(norm_func, axis = 0, arr = dataset)

# get n random values from the 2d source array src_array
def select_n(src_array, n):
  rng = np.random.default_rng()
  return rng.choice(src_array, size = n, replace = False, axis = 0) 
