import numpy as np
import site
from os.path import relpath
from sysconfig import get_paths
from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py
from Cython.Build import cythonize


class build_py(_build_py):
    def run(self):
        self.run_command('build_ext')
        return super().run()

    @staticmethod
    def get_relative_install_path(subdir):
        """Method used to retrieve the relative install path for a compiled library/package.
        The directory to which this path points contains the .cpp file necessary for compilation.

        Parameters
        ----------
        subdir : str
            The sub-directory of the package in which the .cpp file is stored.
        """
        return relpath(f'{site.getsitepackages()[0]}/{subdir}')

    @staticmethod
    def get_relative_header_path(subdir):
        """Method used to retrieve the relative header path for a compiled library/package.
        The directory to which this path points contains the .h file necessary for compilation.

        Parameters
        ----------
        subdir : str
            The subdirectory of the include dirs in which the .h file is stored.
        """
        return relpath(f"{get_paths()['include']}/{subdir}")

    def get_install_paths(self):
        return (
            self.get_relative_install_path('num_integrator'),
            self.get_relative_install_path('ld_funcs'),
            self.get_relative_header_path('num-integrator'),
            self.get_relative_header_path('ld-funcs'),
        )

    def initialize_options(self) -> None:
        super().initialize_options()
        # get the relative install and header paths
        ni_install, ldf_install, ni_head, ldf_head = self.get_install_paths()
        # initialize the external modules distribution variables
        if self.distribution.ext_modules is None:
            self.distribution.ext_modules = []
        # declare package data
        self.distribution.package_data = {
            'cpp-disc-integrator': ['*.pxd'],
            'cpp-disc-integrator/disc_integrator': [
                '*.pxd',
                'disc_integrals.cpp',
            ],
            'cpp-disc-integrator/disc_integrator/libs': ['*.pxd'],
        }
        # declare you want to include these data
        self.distribution.include_package_data = True  # type: ignore
        # declare header files
        self.distribution.headers = [
            f'{ni_head}/num_integration.h',
            f'{ldf_head}/ld_functions.h',
        ]
        # declare you want to include these headers
        self.distribution.include_headers = True  # type: ignore
        # add installation requirements
        self.distribution.install_requires = ['num-integrator', 'ld-funcs']  # type: ignore
        # cythonize/build the external module
        self.distribution.ext_modules = cythonize(
            [
                Extension(
                    'disc_integrator.libcpp_disc_integration',
                    sources=[
                        'disc_integrator/libcpp_disc_integration.pyx',
                        f'{ni_install}/num_integration.cpp',
                        f'{ldf_install}/ld_functions.cpp',
                        'disc_integrator/disc_integrals.cpp',
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
                    include_dirs=[
                        '/usr/local/include',
                        ni_install,
                        ldf_install,
                        ni_head,
                        ldf_head,
                        np.get_include(),
                    ],
                )
            ],
            compiler_directives={'language_level': '3'},
        )
