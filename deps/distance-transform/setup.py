import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

# ----------------------------------------------------------------------------
# Cython Extension
# ----------------------------------------------------------------------------
extensions = [
  Extension('dt',
    ['dt.pyx'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-march=native', '-O3']
  )
]


# ----------------------------------------------------------------------------
# Main Setup
# ----------------------------------------------------------------------------
setup(
  name='Generalized Distance Transforms of Sampled Functions',
  ext_modules = cythonize(extensions),
  test_suite  = 'test'
)
