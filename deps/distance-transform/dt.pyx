import itertools
from numpy cimport ndarray
import numpy as np
cimport numpy as np
cimport cython


# ----------------------------------------------------------------------------
# Distance Function
# ----------------------------------------------------------------------------
cdef class DistanceFunction(object):
  """Interface for defining distance functions

  User-declared distance functions must inherit from this base class so that
  the Cython-compiled code can access the methods provided.
  """
  cdef intersection(self, int x0, int x1, double y0, double y1):
    raise NotImplementedError
  cdef envelope(self, int x, double y):
    raise NotImplementedError


cdef class L2(DistanceFunction):
  """Squared Euclidean distance (L2)

  L2 expresses distance of the form:

    d(p,q) = a*(p - q)^2 + b*(p - q)

  Keyword Args:
    a (float): The quadratic slope (default: 1.0)
    b (float): The quadratic offset (default: 0.0)
  """
  cdef double a, b
  def __init__(self, a=1.0, b=0.0):
    self.a = a
    self.b = b
  cdef intersection(self, int x0, int x1, double y0, double y1):
    return ((y1-y0) - self.b*(x1-x0) + self.a*(x1*x1 - x0*x0)) / (2*self.a*(x1-x0))
  cdef envelope(self, int x, double y):
    return self.a*x*x + self.b*x + y


# ----------------------------------------------------------------------------
# Distance Transform
# ----------------------------------------------------------------------------
def compute(x, axes=None, f=L2):
  """Compute the distance transform of a sampled function

  Compute the N-dimensional distance transform using the method described in:

    P. Felzenszwalb, D. Huttenlocher "Distance Transforms of Sampled Functions"

  Args:
    x (ndarray): An n-dimensional array representing the data term

  Keyword Args:
    axes (tuple): The axes over which to perform the distance transforms. The
      order does not matter. (default all axes)
    f (DistanceFunction): The distance function to apply (default L2)
  """
  shape = x.shape
  axes = axes if axes else tuple(range(x.ndim))
  f = f() if isinstance(f, type) else f

  # initialize the minima and argument arrays
  min = x.copy()
  arg = tuple(np.empty(shape, dtype=int) for axis in axes)

  # create some scratch space for the transforms
  v = np.empty((max(shape)+1,), dtype=int)
  z = np.empty((max(shape)+1,), dtype=float)

  # compute transforms over the given axes
  for n, axis in enumerate(axes):

    numel  = shape[axis]
    minbuf = np.empty((numel,), dtype=float)
    argbuf = np.empty((numel,), dtype=int)
    slices = map(xrange, shape)
    slices[axis] = [Ellipsis]

    for index in itertools.product(*slices):

      # compute the optimal minima
      _compute1d(min[index], f, minbuf, argbuf, z, v)
      min[index] = minbuf
      arg[n][index] = argbuf
      nindex = tuple(argbuf if i is Ellipsis else i for i in index)

      # update the optimal arguments across preceding axes
      for m in reversed(range(n)):
        arg[m][index] = arg[m][nindex]

  # return the minimum and the argument
  return min, arg


# ----------------------------------------------------------------------------
# 1D Distance Transform (Cython)
# ----------------------------------------------------------------------------
@cython.boundscheck(False)
cdef _compute1d(
    ndarray[double] x, DistanceFunction f,  # input array and distance function
    ndarray[double] min, ndarray[long] arg, # output arrays
    ndarray[double] z, ndarray[long] v):    # working buffers
  """Low-level 1D distance transform

  This Cython function provides the implementation of the 1D distance transform.
  It is compiled for speed - it is roughly 150x faster than the same Python
  implementation without type declarations. It optimizes:

    arg min f(p,q) + x(q)
         q

  Args:
    x (ndarray): The input
    f (DistanceFunction): The distance function
    min (ndarray): The minimum solution
    arg (ndarray): The argument of the minimum
    z (ndarray): A double-precision working buffer of length N+1
    v (ndarray): An integer-precision working buffer of length N
  """

  # predeclare object types
  cdef int N = x.shape[0]
  cdef int k, q
  cdef double s
  z.fill(np.inf)

  # initial conditions
  v[0], z[0] = 0, -np.inf

  # compute the intersection points
  k = 0
  for q in xrange(1,N):
    s = f.intersection(v[k], q, x[v[k]], x[q])
    while s <= z[k]:
      k = k-1
      s = f.intersection(v[k], q, x[v[k]], x[q])
    k, v[k], z[k] = k+1, q, s

  # compute the projection onto the lower envelope
  k = 0
  for q in xrange(N):
    while z[k+1] < q: k += 1
    min[q], arg[q] = f.envelope(q-v[k], x[v[k]]), v[k]
