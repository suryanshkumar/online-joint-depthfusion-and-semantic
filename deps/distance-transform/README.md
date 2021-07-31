Distance Transform
==================

Compute the generalized distance transform of a sampled function

Introduction
------------

This module provides a Python implementation of the linear-time distance transform described in:

  > P. Felzenszwalb, D. Huttenlocher "Distance Transforms of Sampled Functions"

Computing the distance transform is as easy as:

```python
import dt
import numpy as np

x = np.random.standard_normal((100,100))
y,i = dt.compute(x)
```

  This module can handle arbitrary dimensional data:

```python
x = np.random.standard_normal((100,100,4,5))
y,i = dt.compute(x)  # compute the distance transform across ALL dimensions
y,i = dt.compute(x, axes=(0,1)) # Compute across the (0,1) axes in the tensor
```

You can also change the distance function, or parameters used:

```python
y,i = dt.compute(x, f=dt.L2(0.01)) # reduce the distance penalty
```

Installing
----------

Install the package using pip:

    pip install git+https://github.com/hbristow/distance-transform

You will need Cython to build the extensions.
