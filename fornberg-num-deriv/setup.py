from setuptools import setup
import os
import re
from fornberg_num_deriv import __version__


# local definitions shortening commands
fm_path = './fornberg_num_deriv/'
fnfile = 'fornberg.f90'
fnwrap = 'f90wrap_fornberg.f90'
modname = 'fnd'
opts = '--f90flags="-O3"'


# compile a regex to aid with library file detection
rcomp = re.compile(rf'^{modname}\.cpython.*\.so')


# compile the fortran / python library
print('\n\nCompiling Fortran f2py wrapper module.\n\n')
shared_obj_comp = f'python -m numpy.f2py -c -m {modname} {fm_path}{fnfile} {fm_path}{fnwrap} {opts}'
print('\n' + shared_obj_comp + '\n')
os.system(shared_obj_comp)
print('\n\nFortran f2py3 wrapper compiled.\n\n')


# find the name of the shared library you just built
libname = [x.name for x in os.scandir('.') if rcomp.match(x.name)][0]


# make the library accessible
setup(
    packages=['myfnd'],
    package_dir={'myfnd': '.'},
    package_data={'myfnd': [libname]},
    version=__version__,
)
