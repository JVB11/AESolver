# cpp-disc-integrator

Interfacing library between C++ functions that compute disc integral factors defined in e.g. Van Beeck et al. (forthcoming) and Python, using Cython.

## Installation notes

Install this package locally in a directory that is a python 'includedir', by activating your (favorite) python virtual environment and running the command 'pip install cpp-disc-integrator'.
(Optionally one might need to add the option '--use-feature=in-tree-build', depending on what version of pip is used to install the package.)

> Note: the installation of this package requires several Python packages to be installed. The Python package dependencies are:
>
> - cpp-ld-funcs / ld_funcs
> - cpp-numintegrator / num_integrator
> - [Cython](https://cython.org/)
>
> In addition, we also require [OpenMP](https://www.openmp.org/) to be installed, so that the C++ modules/libraries can be compiled and parallelized methods can be called/used.
>
> Should one wish to run the C++ Google tests (using ['cpp_test.sh'](cpp_test.sh)), one also needs to have [Cmake](https://cmake.org/) and [googletest](https://github.com/google/googletest) installed.

## Python import notes

To import this package into a Python module/script after having installed it in your (favorite) Python virtual environment, one should write

```python
from disc_integrator import di
```

where 'di' is the module that contains the method 'get_integrals', which computes the first and second disc integral factors defined in e.g. Van Beeck et al. (forthcoming).

## Usage notes

The module 'di' can be used to compute the disc integral factors defined in e.g. Van Beeck et al. (forthcoming), based on(numpy) arrays of mu = cos(theta) values, angular function values and values of the first and second mu derivatives of these angular functions.
For example, one can write

```python
# import the relevant packages after installation of the 'disc_integrator' package
from disc_integrator import di
import numpy as np  # import to generate array
from numpy.random import default_rng  # import random number generator


# set up three random number generators containing a specific seed
rng_1 = default_rng(seed=41)
rng_2 = default_rng(seed=42)
rng_3 = default_rng(seed=43)


# generate input arrays
my_mus = np.linspace(0.0, 1.0, num=10)  # mu = cos(theta) values
my_ang_func = rng_1.random((10,))  # random values for angular function
my_first_mu_der = rng_2.random((10,))  # same for first derivative
my_second_mu_der = rng_3.random((10,))  # same for second derivative


# generate Python input variables, for the sake of explaining how one should provide input for the 'compute' function
my_array_size = my_mus.shape[0]  # array size
my_number_of_openmp_threads = 4  # set the number of openmp threads to be used,
use_parallelized_version = False  
# determine whether to use parallelized numerical integration (or serial)
ld_function = "eddington"  # determines which limb-darkening function is computed.
# The only option for now is "eddington", which computes the limb-darkening function as 1.0 + 1.5 * my_mus. 
integration_method = "trapz"  # determines which numerical integration method is used. 
# Options are "trapz" and "simpson". 
# Note that "simpson" is only implemented for equally spaced integration variables, and will use the "trapz" algorithm if 'equally_spaced' is False.


# compute the disc integral factors and store the result in (unpacked) variables  (function return a tuple of values that can be unpacked on the fly)
first_disc, second_disc = di.get_integrals(
    my_mus, my_ang_func, my_first_mu_der,
    my_second_mu_der, ld_function, integration_method,
    my_number_of_openmp_threads, use_parallelized_version
)


# print the disc integral factors
print(f"The first disc integral factor is '{first_disc:.15f}'.")
print(f"The second disc integral factor is '{second_disc:.15f}'.")
```

Running the example should print

```sh
The first disc integral factor is '0.5236444131194912'.
The second disc integral factor is '0.52551432999423'.
```

which is the correct result up to 1e-16 (i.e. of the magnitude of a floating point error), as can be verified by running the corresponding script, which computes the disc integral factors using Numpy,

```python
# import the relevant packages
import numpy as np
from numpy.random import default_rng


# set up three random number generators containing a specific seed
rng_1 = default_rng(seed=41)
rng_2 = default_rng(seed=42)
rng_3 = default_rng(seed=43)


# generate input arrays
my_mus = np.linspace(0.0, 1.0, num=10)  # mu = cos(theta) values
my_ang_func = rng_1.random((10,))  # random values for angular function
my_first_mu_der = rng_2.random((10,))  # same for first derivative
my_second_mu_der = rng_3.random((10,))  # same for second derivative


# ensure mus (and corresponding data arrays) are in the 0 to 1 region 
# --> enforce integration boundaries
def enforce_integration_boundaries(mus):
    # generate and return mask
    return (mus >= 0.0) & (mus <= 1.0)


# define a function that computes the Eddington ld function
def compute_eddington_ld_function(mus):
    return 1.0 + (1.5 * mus)


# define a function that computes the first disc integral factor
def compute_first_disc(mus,ang):
    # compute mask
    mm = enforce_integration_boundaries(mus)
    # compute ldf
    ldf = compute_eddington_ld_function(mus)
    # compute the integrand
    integrand = mus[mm] * ang[mm] * ldf[mm]
    # compute the integral
    return np.trapz(integrand, mus[mm])


# define a function that computes the second disc integral factor
def compute_second_disc(mus,fder,sder):
    # compute mask
    mm = enforce_integration_boundaries(mus)
    # compute ldf
    ldf = compute_eddington_ld_function(mus)
    # compute the integrand
    integrand = 2.0 * (mus[mm]**2.0) * fder[mm] * ldf[mm]
    integrand -= (mus[mm] - mus[mm]**3.0) * sder[mm] * ldf[mm]
    # compute the integral
    return np.trapz(integrand, mus[mm])


# compute the disc integral factors
first_disc = compute_first_disc(my_mus, my_ang_func)
second_disc = compute_second_disc(my_mus, my_first_mu_der, my_second_mu_der)


# print the disc integral factors
print(f"The first disc integral factor is '{first_disc:.15f}'.")
print(f"The second disc integral factor is '{second_disc:.15f}'.")
```

For additional information on these specific modules you may consult the [API documentation](https://jvb11.github.io/AESolver/overview_API/API_cpp_disc_integrator/API_index.html).

### Author: Jordan Van Beeck
