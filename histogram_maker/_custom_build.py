import numpy as np
from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py
from Cython.Build import cythonize


class build_py(_build_py):
    def run(self):
        self.run_command('build_ext')
        return super().run()

    def initialize_options(self) -> None:
        super().initialize_options()
        if self.distribution.ext_modules is None:
            self.distribution.ext_modules = []
        # declare package data
        self.distribution.package_data = {
            'hogg_jacky': ['*.pxd'],
            'hogg_jacky/cython_jacky': ['*.pxd'],
            'hogg_jacky/cython_jacky/lib': ['*.pxd'],
        }
        # declare you want to include these data
        self.distribution.include_package_data = True  # type: ignore
        # declare header files
        self.distribution.headers = []
        # cythonize/build the external module
        self.distribution.ext_modules = cythonize(
            [
                Extension(
                    'hogg_jacky.cython_jacky.cython_jacky_binning',
                    sources=[
                        'hogg_jacky/cython_jacky/cython_jacky_binning.pyx'
                    ],
                    extra_compile_args=['-O3', '-fopenmp'],
                    extra_link_args=['-O3', '-fopenmp'],
                    include_dirs=[np.get_include()],
                ),
                Extension(
                    'hogg_jacky.cython_jacky.cython_jacky_empirical_probability_conversion_factor',
                    sources=[
                        'hogg_jacky/cython_jacky/cython_jacky_empirical_probability_conversion_factor.pyx'
                    ],
                    extra_compile_args=['-O3', '-fopenmp'],
                    extra_link_args=['-O3', '-fopenmp'],
                    include_dirs=[np.get_include()],
                ),
                Extension(
                    'hogg_jacky.cython_jacky.cython_jacky_bin_widths',
                    sources=[
                        'hogg_jacky/cython_jacky/cython_jacky_bin_widths.pyx'
                    ],
                    extra_compile_args=['-O3', '-fopenmp'],
                    extra_link_args=['-O3', '-fopenmp'],
                    include_dirs=[np.get_include()],
                ),
                Extension(
                    'hogg_jacky.cython_jacky.cython_jacky_jacky',
                    sources=['hogg_jacky/cython_jacky/cython_jacky_jacky.pyx'],
                    extra_compile_args=['-O3', '-fopenmp'],
                    extra_link_args=['-O3', '-fopenmp'],
                    include_dirs=[np.get_include()],
                ),
                Extension(
                    'hogg_jacky.cython_jacky.cython_jacky_empirical_probability_hist',
                    sources=[
                        'hogg_jacky/cython_jacky/cython_jacky_empirical_probability_hist.pyx'
                    ],
                    extra_compile_args=['-O3', '-fopenmp'],
                    extra_link_args=['-O3', '-fopenmp'],
                    include_dirs=[np.get_include()],
                ),
                Extension(
                    'hogg_jacky.cython_jacky.cython_jacky_empirical_density_hist',
                    sources=[
                        'hogg_jacky/cython_jacky/cython_jacky_empirical_density_hist.pyx'
                    ],
                    extra_compile_args=['-O3', '-fopenmp'],
                    extra_link_args=['-O3', '-fopenmp'],
                    include_dirs=[np.get_include()],
                ),
            ],
            compiler_directives={'language_level': '3'},
        )
