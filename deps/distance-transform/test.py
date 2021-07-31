import dt
import unittest
import numpy as np

# ----------------------------------------------------------------------------
# SYMMETRY
# ----------------------------------------------------------------------------
class TestDT(unittest.TestCase):

  def test_identity(self):
    """Assert that equal potentials are already ground truth"""
    x = np.ones((10,13))
    xs,i = dt.compute(x)
    self.assertTrue((xs == x).all())

  def test_independence(self):
    """Assert the the order of transforms along axes does not matter"""
    x = np.random.standard_normal((10,11,3))
    xs1,i1 = dt.compute(x, axes=(0,1,2))
    xs2,i2 = dt.compute(x, axes=(2,1,0))

    self.assertTrue(np.linalg.norm(xs1 - xs2) == 0.0)
    self.assertTrue(np.linalg.norm(i1[0] - i2[2]) == 0.0)

  def test_distance_cost(self):
    """Assert that the minimum solution cost > the minimum potential"""
    for n in xrange(100):
      x = np.random.standard_normal((9,13))
      xs,i = dt.compute(x)
      self.assertTrue(xs.min() >= x.min())
