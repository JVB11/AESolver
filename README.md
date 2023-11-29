[![GitHub issues](https://img.shields.io/github/issues/JVB11/AESolver)](https://github.com/JVB11/AESolver/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/JVB11/AESolver)](https://github.com/JVB11/AESolver/pulls)
[![GitHub watchers](https://img.shields.io/github/watchers/JVB11/AESolver)](https://github.com/JVB11/AESolver/watchers)
[![GitHub forks](https://img.shields.io/github/forks/JVB11/AESolver)](https://github.com/JVB11/AESolver/forks)
[![GitHub Repo stars](https://img.shields.io/github/stars/JVB11/AESolver)](https://github.com/JVB11/AESolver/stargazers)
![GitHub License](https://img.shields.io/github/license/jvb11/aesolver)
![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/JVB11/AESolver)
<!-- ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJVB11%2FAESolver%2Fmain%2Faesolver%2Fpyproject.toml)
![Lines of code](https://img.shields.io/tokei/lines/github/JVB11/AESolver) -->

<p align="center">
  <img src="/logo/AE_solver_logo.png" alt="drawing" width="500"/>
</p>

# AESolver (Amplitude Equation Solver)

![Ubuntu](https://img.shields.io/badge/Ubuntu-Tested-black?style=for-the-badge&logo=ubuntu&logoColor=white&labelColor=E95420)
![MacOs](https://img.shields.io/badge/mac%20os-In_progress-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)

The coupled-mode and amplitude equations (AEs) are generic equations of motion that describe the coupling among various stellar oscillation modes.

The Python packages in this repository allow the user to compute solutions of the AEs. These solutions can be time-dependent or stationary.
Such solutions can be used to generate observables that can be compared to detected (non-linear) stellar oscillation signals in (space) photometry.

For your convenience, we provide scripts that allow you to set up your own private Python environment that contains the necessary packages (we support a [Conda](https://docs.conda.io/en/latest/)/[Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) install for now).
We also provide run scripts that can be used to perform the various computations, where the run parameters are specified in an inlist.

Because we use [Fortran](https://fortran-lang.org/) and [C++](https://isocpp.org/) modules to do some of the heavy lifting, a working C++ and Fortran compiler (you can for example use [the gnu compilers](https://gcc.gnu.org)) are required to generate  Python-interfaced wheels. These wheels perform the number crunching using the compiled language modules (written in C++ or Fortran).

Author: Jordan Van Beeck <a href="https://orcid.org/0000-0002-5082-3887"> <img alt="ORCID logo" src="https://orcid.figshare.com/ndownloader/files/8439047/preview/8439047/preview.jpg" width="16" height="16"/></a>

![Python](https://img.shields.io/badge/python_powered-3670A0?style=for-the-badge&logo=python&logoColor=white)
![C++](https://img.shields.io/badge/c++_powered-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Fortran](https://img.shields.io/badge/Fortran_powered-%23734F96.svg?style=for-the-badge&logo=fortran&logoColor=white)

![Matplotlib](https://img.shields.io/badge/Matplotlib_powered-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Numpy](https://img.shields.io/badge/numpy_powered-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas_powered-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy_powered-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)
