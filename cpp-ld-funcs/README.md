# cpp-ld-funcs

Interfacing library between C++ functions that compute limb-darkening functions and Python, using Cython.

## Installation notes

Install this package locally in a directory that is a python 'includedir', by activating your (favorite) python virtual environment and running the command 'pip install cpp-ld-funcs'.
(Optionally one might need to add the option '--use-feature=in-tree-build', depending on what version of pip is used to install the package.)

> Note: the installation of this package requires [Cython](https://cython.org/) to be installed (in your favorite virtual environment). In addition, we also require [OpenMP](https://www.openmp.org/) to be installed, so that the C++ modules/libraries can be compiled and parallelized methods can be called/used.
>
> Should one wish to run the C++ Google tests (using ['cpp_test.sh'](cpp_test.sh)), one also needs to have [Cmake](https://cmake.org/) and [googletest](https://github.com/google/googletest) installed.

## Python import notes

To import this package into a Python module/script after having installed it in your (favorite) Python virtual environment, one should write

```python
from ld_funcs import ldf
```

where 'ldf' is the module that contains the method 'compute' that computes the limb-darkening function requested.

## Usage notes

The module 'ldf' can be used to compute a limb-darkening function based on a (numpy) array of mu = cos(theta) values.
For example, one can write

```python
# import the relevant packages after installation of the 'ld_funcs' package
from ld_funcs import ldf
import numpy as np  # import to generate array
# generate array of mu = cos(theta) values
my_mus = np.linspace(0.0, 1.0, num=10)
# generate Python input variables, for the sake of explaining
# how one should provide input for the 'compute' function
my_array_size = my_mus.shape[0]  # array size
my_number_of_openmp_threads = 4  # set the number of openmp threads to be used,
use_parallelized_version = False  
# determine whether to use parallelized numerical integration (or serial)
ld_function = "eddington"  # determines which limb-darkening function is computed.
# The only option for now is "eddington", which computes the limb-darkening function
# as 1.0 + 1.5 * my_mus. 
my_limb_darkening_function = np.zeros_like(my_mus)  # will contain the limb darkening function
# compute the limb-darkening function and store the result in a variable
ldf.compute(
    my_mus, my_limb_darkening_function, ld_function,
    my_number_of_openmp_threads, use_parallelized_version
)
# print the limb-darkening function values
print(f"Eddington ldf vals are '{my_limb_darkening_function.tolist()}'.")
```

Running the example should print

```sh
Eddington ldf vals are '[1.0, 1.1666666666666667, 1.3333333333333333, 1.5, 1.6666666666666665, 1.8333333333333335, 2.0, 2.1666666666666665, 2.333333333333333, 2.5]'. 
```

which is the correct result, as can be verified by running the corresponding function on the input array, e.g.

```python
print(f"{1.0 + 1.5 * my_mus}")
```

For additional information on these specific modules you may consult the [API documentation](https://jvb11.github.io/AESolver/overview_API/API_cpp_ld_funcs/API_index.html).

### Author: Jordan Van Beeck
