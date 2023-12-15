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
            'cpp-numintegrator': ['*.pxd'],
            'cpp-numintegrator/num_integrator': [
                '*.pxd',
                'num_integration.cpp',
            ],
            'cpp-numintegrator/num_integrator/libs': ['*.pxd'],
        }
        # declare you want to include these data
        self.distribution.include_package_data = True  # type: ignore
        # declare header files
        self.distribution.headers = ['num_integrator/num_integration.h']
        # cythonize/build the external module
        self.distribution.ext_modules = cythonize(
            [
                Extension(
                    'num_integrator.libcpp_num_integrator',
                    sources=[
                        'num_integrator/libcpp_num_integrator.pyx',
                        'num_integrator/num_integration.cpp',
                    ],
                    extra_compile_args=[
                        '-std=c++17',
                        '-O3',
                        '-fopenmp',
                        '-fPIC',
                        '-shared',
                    ],
                    extra_link_args=['-fPIC', '-fopenmp', '-shared'],
                    language='c++',
                    include_dirs=[np.get_include()],
                )
            ],
            compiler_directives={'language_level': '3'},
        )
