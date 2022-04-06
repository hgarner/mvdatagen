import unittest

import data_generation_mvnorm as dg
import numpy as np

class TestMvNorm(unittest.TestCase):

  def test_gen_corr_matrix(self):
    # todo
    # test exceed num_vars limits for cols
    # test multiple col selection
    # test incorrect types
    
    num_vars = 12
    neg_cols = (0, 3)
    pos_cols = (3, 9)
    neg_range = (-1, -0.6)
    pos_range = (0.6, 1)
    mid_range = (0, 0.3)

    corr = dg.gen_corr_matrix(
      num_vars = num_vars,
      neg_cols = neg_cols,
      pos_cols = pos_cols,
      neg_range = neg_range,
      pos_range = pos_range,
      mid_range = mid_range
    )

    # test that corr is ndarray
    self.assertIsInstance(corr, np.ndarray, 'matrix must be of type numpy.ndarray')

    # test that a matrix of the correct size is generated
    self.assertEqual(np.shape(corr), (12, 12), 'incorrect shape')

    # test that the diagonals are all 1
    self.assertTrue(np.allclose(np.diag(corr), 1.0), 'matrix must have diagonals of 1')

    # test +ve definite
    w, v = np.linalg.eig(corr)
    self.assertTrue((w > 0).all(), 'matrix must have +ve eigenvalues')
    self.assertTrue(np.allclose(corr, corr.T), 'matrix must be symmetric')

if __name__ == '__main__':
  unittest.main()
