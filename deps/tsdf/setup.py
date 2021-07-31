from Cython.Build import cythonize

from setuptools import setup, find_packages
from setuptools.extension import Extension

import numpy as np

ext_modules = [
    Extension("tsdf.TSDFVolume", ["src/tsdf/TSDFVolume.pyx"],
              include_dirs=[np.get_include()])
]

setup(name='tsdf',
      version='0.1dev',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      ext_modules=cythonize(ext_modules))
