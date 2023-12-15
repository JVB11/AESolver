# fornberg-num-deriv

Python wrapper for the Fortran code that implements the numerical derivation method of Fornberg (1988), as implemented on [Cococubed](https://cococubed.com/code_pages/fdcoef.shtml) in Fortran.
Specifically, this module contains:

1. the myfnd module that contains the Fortran wrapper called 'fnd'.

## REQUIRES

python modules:

1. meson
2. ninja

## COMPILE

To compile this Fortran module interfaced with python, use

```console
pip install 'path to directory'
```

This should install the 'myfnd' python package.

## USE

A test script is available that shows how to use this module.
Please check and run

```console
python test.py
```

in the git dir of this repository to test out the module.

## UNINSTALL

Removing the installed package can be done by running

```console
pip uninstall myfnd
```

For additional information on these specific modules you may consult the [API documentation](https://jvb11.github.io/AESolver/overview_API/API_fornberg_num_deriv/API_index.html).

### Author: Jordan Van Beeck
