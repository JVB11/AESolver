[![GitHub issues](https://img.shields.io/github/issues/JVB11/AESolver)](https://github.com/JVB11/AESolver/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/JVB11/AESolver)](https://github.com/JVB11/AESolver/pulls)
[![GitHub watchers](https://img.shields.io/github/watchers/JVB11/AESolver)](https://github.com/JVB11/AESolver/watchers)
[![GitHub forks](https://img.shields.io/github/forks/JVB11/AESolver)](https://github.com/JVB11/AESolver/forks)
[![GitHub Repo stars](https://img.shields.io/github/stars/JVB11/AESolver)](https://github.com/JVB11/AESolver/stargazers)
[![GitHub License](https://img.shields.io/github/license/jvb11/aesolver)](./LICENSE)
[![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/JVB11/AESolver)](https://github.com/JVB11/AESolver/commits/main)

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

Because we use [Fortran](https://fortran-lang.org/) and [C++](https://isocpp.org/) modules to do some of the heavy lifting, a working C++ and Fortran compiler (you can for example use [the gnu compilers](https://gcc.gnu.org)) are required to generate  Python-interfaced wheels (using [Cython](https://cython.org/)). These wheels perform the number crunching using the compiled language modules (written in C++ or Fortran and interfaced with Cython).

### When using (part of) this code, you should cite the following works:

1) [Van Beeck, J., Van Hoolst, T., Aerts, C., \& Fuller, J. (2023](https://ui.adsabs.harvard.edu/abs/2023arXiv231102972V/abstract); under revision at A\&A; manuscript available at [https://arxiv.org/abs/2311.02972](https://arxiv.org/abs/2311.02972)).

### Author: Jordan Van Beeck <a href="https://orcid.org/0000-0002-5082-3887"> <img alt="ORCID logo" src="./logo/ORCID-iD_icon_16x16.png" width="16" height="16"/></a>

![Python](https://img.shields.io/badge/python_powered-3670A0?style=for-the-badge&logo=python&logoColor=white)
![C++](https://img.shields.io/badge/c++_powered-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Fortran](https://img.shields.io/badge/Fortran_powered-%23734F96.svg?style=for-the-badge&logo=fortran&logoColor=white)
![Cython](https://img.shields.io/badge/Cython_powered-%23ffffff.svg?style=for-the-badge&logo=data:image/svg%2bxml;base64,AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABkZGQBZGRkS2RkZKNkZGTaZGRk9mRkZPxkZGTtZGRkxmRkZINkZGQgAAAAAAAAAAAAAAAAAAAAAAAAAABkZGQfZGRkxGRkZP9kZGT/ZWVl12pqaqNra2uaZmZmq2RkZOlkZGT/ZGRk+WRkZIJkZGQDAAAAAAAAAABkZGQgZGRk5GRkZP9kZGT5ZGRkXoiIiAOampoIm5ubCY+PjwRlZWUHZGRkjGRkZP9kZGT/ZGRko2RkZAFkZGQCZGRkymRkZP9kZGT/ZGRkbwAAAABR2/8/Rtj/wj3U/9s71P+RO9T/FmRkZABkZGSQZGRkx2RkZMdkZGRJZGRkV2RkZP9kZGT/ZGRk4WRkZAUAAAAAXeD/uVTc//9J2f//Qdb/njvU/2YAAAAAAAAAAAAAAAAAAAAAAAAAAGRkZLFkZGT/ZGRk/2RkZJCpeDw7o3M4nG3k/bxj4v//Wd7/007b/7xE1/+qPNT/lTvU/xMAAAAAAAAAAAAAAABkZGTmZGRk/2RkZP9kZGRgr35Azah3O/923vC5cOb//2jk//9d4P//Udz//0bY//8/1f98AAAAAAAAAAAAAAAAZGRk/GRkZP9kZGT/ZGRkTbWDRO6ufT//o4NPy4uplbyGppS9fqufumHf/OJW3f//Tdr/ngAAAAAAAAAAAAAAAGRkZPRkZGT/ZGRk/2RkZFO7iEjJtINE/618P/+mdTr/n281/5lqMf+BqZu6ZeP//1zg/3kAAAAAAAAAAAAAAABkZGTMZGRk/2RkZP9kZGRvvotKLbmHR5GzgUKyrHs+vKR0OeiebjT/jJJ0mnDm/41p5P8SAAAAAAAAAAAAAAAAZGRke2RkZP9kZGT/ZGRktAAAAAAAAAAAuYZHobF/QbGrej3/pHQ4/59vNGYAAAAAZGRkC2RkZCFkZGQhZGRkH2RkZBBkZGTnZGRk/2RkZP1kZGQ2AAAAAL2KSk64hUassX9Byql4PKejczgVZGRkAWRkZLJkZGT/ZGRk/2RkZKMAAAAAZGRkPmRkZPZkZGT/ZGRk6GRkZEEAAAAAAAAAAAAAAAAAAAAAZGRkDWRkZJxkZGT/ZGRk/2RkZNBkZGQPAAAAAAAAAABkZGQ5ZGRk32RkZP9kZGT/ZGRkzmRkZJRkZGSGZGRkqWRkZO9kZGT/ZGRk/2RkZLBkZGQRAAAAAAAAAAAAAAAAAAAAAGRkZApkZGRuZGRkx2RkZPpkZGT/ZGRk/2RkZP9kZGTuZGRkq2RkZERkZGQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABkZGQGZGRkHmRkZCVkZGQWZGRkAQAAAAAAAAAAAAAAAAAAAAAAAAAA//8AAPPvAADv8wAAz/8AAJ5/AACe/wAAmg8AAJvvAACYLwAAn78AAJ8/AADf+QAA7/MAAPPnAAD+PwAA//8AAA==)

![Matplotlib](https://img.shields.io/badge/Matplotlib_powered-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Numpy](https://img.shields.io/badge/numpy_powered-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas_powered-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy_powered-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)
