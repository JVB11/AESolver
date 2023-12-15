# cpp-numintegrator

Numerical integration methods implemented in C++, and interfaced with Python using Cython.

## Installation notes

Install this package locally in a directory that is a Python 'includedir', by activating your (favorite) Python virtual environment and running the command 'pip install cpp-numintegrator'.
(Optionally one might need to add the option '--use-feature=in-tree-build', depending on what version of pip is used to install the package.)

> Note: the installation of this package requires [Cython](https://cython.org/) to be installed (in your favorite virtual environment). In addition, we also require [OpenMP](https://www.openmp.org/) to be installed, so that the C++ modules/libraries can be compiled and parallelized numerical integration methods can be called/used.
>
> Should one wish to run the C++ Google tests (using ['cpp_test.sh'](cpp_test.sh)), one also needs to have [Cmake](https://cmake.org/) and [googletest](https://github.com/google/googletest) installed.

## Python import notes

To import this package into a Python module/script after having installed it in your (favorite) Python virtual environment, one should write

```python
from num_integrator import cni
```

where 'cni' is the module that contains the method 'integrate' that performs numerical integration.

## Usage notes

The module 'cni' can be used to numerically integrate a function based on a (numpy) array of integrand values and a (numpy) array of integration variables.
For example, one can write

```python
# import the relevant packages after installation of the 'num_integrator' package
from num_integrator import cni
import numpy as np  # imported to generate arrays


# generate integrand and integration variable arrays
my_integrand_vals = np.array([1.0, 2.0, 3.0])
my_integration_variable_vals = np.array([0.02, 0.07, 0.12])


# generate Python input variables, for the sake of explaining
# how one should provide input for the 'integrate' function
my_array_size = my_integrand_vals.shape[0]  # array size
my_number_of_openmp_threads = 4  # set the number of openmp threads to be used,
# if parallelized numerical integration is required
use_parallelized_version = False  
# determine whether to use parallelized numerical integration (or serial)
integration_method = "trapz"  # determines which numerical integration method is used. 
# Options are "trapz" and "simpson". 
# Note that "simpson" is only implemented for equally spaced integration variables, 
# and will use the "trapz" algorithm if 'equally_spaced' is False.
equally_spaced_integration_variable = True  # define whether the integration variables 
# are equally spaced (which is True in this example)


# perform the numerical integration and store the result in a variable
my_numerical_integration_result = cni.integrate(
    integration_method, equally_spaced_integration_variable,
    my_integrand_vals, my_integration_variable_vals,
    my_number_of_openmp_threads, use_parallelized_version
)


# print the numerical integration result
print(f"Numerical integration using the {integration_method} method "
      f"yielded '{my_numerical_integration_result:.16f}'.") 
```

Running the example should print

```sh
Numerical integration using the trapz method yielded '0.2000000000000000'. 
```

which is the correct result up to an order of magnitude of 1e-17 (i.e. of the order of magnitude of floating point errors), as can be verified by running the corresponding numpy function on the input arrays, e.g.

```python
print(f"{np.trapz(my_integrand_vals, my_integration_variable_vals):.16f}")
```

For additional information on these specific modules you may consult the [API documentation](https://jvb11.github.io/AESolver/overview_API/API_cpp_numintegrator/API_index.html).

### Author: Jordan Van Beeck
