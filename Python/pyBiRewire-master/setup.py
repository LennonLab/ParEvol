## This code is written by Andrea Gobbi, <gobbi.andrea@mail.com>.
## (C) 2015 BiRewire Developers.

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

NAME = "BiRewire"
VERSION = "0.6"
DESCR = "A Cython wrapper of BiRewire"
URL = "http://bioconductor.org/packages/devel/bioc/html/BiRewire.html"
REQUIRES = ['numpy', 'cython','igraph','jgraph']

AUTHOR = "Andrea Gobbi"
EMAIL = "gobbi.andrea@mail.com"

LICENSE = "GPL V3"

SRC_DIR = "BiRewire"
PACKAGES = [SRC_DIR]

ext_1 = Extension(SRC_DIR + ".wrapped",
                  [SRC_DIR + "/lib/BiRewire.c", SRC_DIR + "/wrapped.pyx"],
                  libraries=[],
                  include_dirs=[np.get_include()])


EXTENSIONS = [ext_1]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=EXTENSIONS
          )
