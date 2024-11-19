from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

ext_path = Path("huang")

ext_modules = [
    Extension(
        "loops",
        [str(ext_path / "loops.pyx")],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="huang",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)
