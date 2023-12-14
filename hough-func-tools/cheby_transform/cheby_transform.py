"""Python module that provides a collocation framework to compute the Chebyshev collocation solution of a differential equation.

Notes
-----
An earlier version of this module was used by Prat et al. (2019) to compute the Hough functions from the Laplace Tidal equations, using a collocation method similar to Wang et al. (2016).

License: GPL-3+
Author: Vincent Prat <vincent.prat@cea.fr>

Code modifications by Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np


# Chebyshev transform class
class ChebyTransform:
    """Class defining the methods used to perform Chebyshev collocation in order to solve a differential equation.

    Parameters
    ----------
    npts : int, optional
        The total number of points used for the collocation. This has to be an even number; by default 200.
    """

    # attribute declarations
    matrix_size: int
    npts: int
    theta: np.ndarray
    mu: np.ndarray
    s: np.ndarray
    d2: np.ndarray
    d1: np.ndarray
    d0: np.ndarray

    # initialization method
    def __init__(self, npts: int = 200) -> None:
        # enforce an even number of points
        self.m_size = npts // 2
        self.npts = self.m_size * 2  # (axi)symmetric
        # Calculate the interior/root points (mu_i = cos(((2i-1)Pi)/2N), where i = 1,...,N , with N = total number of collocation points; see Wang et al. 2016)
        self._compute_interior_root_points()
        # Calculate the sin/cos (theta) factors
        self._compute_sin_cos_factors()
        # Calculate the Chebyshev matrix (see Wang et al. 2016)
        self._compute_chebyshev_matrix()
        # multiply derivatives with inverse of T_n (d0) -> obtain unity matrix coeffs0 accompanying term (see Full matrix projection below)
        self._prepare_full_expansion()

    def _compute_interior_root_points(self) -> None:
        """Initialization method used to compute the interior/root points."""
        # compute values of theta
        n = np.arange(self.m_size)
        theta = np.pi / self.npts * (n + 0.5)
        self.theta = theta

    def _compute_sin_cos_factors(self) -> None:
        """Initialization method used to compute sin(theta)/cos(theta) values."""
        # compute additional variables:
        # --- mu = cos(theta)
        mu = np.cos(self.theta)
        self.mu = mu
        # --- s = sin(theta)
        s = np.sqrt(1 - mu**2)
        self.s = s

    def _compute_chebyshev_matrix(self) -> None:
        """Initialization method used to compute the Chebyshev matrix."""
        # Initialize the coefficient result Numpy arrays
        self.d2 = np.zeros((2, 2, self.m_size, self.m_size))
        self.d1 = np.zeros_like(self.d2)
        self.d0 = np.zeros_like(self.d2)

        # d0 = chebyshev polynomial T_n (/times pf) ; d1 = first derivative with respect to mu ; d2 = second derivative with respect to mu; see also Boyd (2001)
        for i in range(self.m_size):
            for j in range(self.m_size):
                # even parity
                j_index = 2 * j
                cij = np.cos(
                    np.pi * j_index / self.npts * (self.npts + i + 0.5)
                )
                sij = np.sin(
                    np.pi * j_index / self.npts * (self.npts + i + 0.5)
                )

                # with parity factor
                self.d0[0, 1, i, j] = cij * self.s[i]
                self.d1[0, 1, i, j] = (
                    j_index * sij - cij * self.mu[i] / self.s[i]
                )
                self.d2[0, 1, i, j] = (
                    -(j_index**2) * cij * self.s[i] ** 2
                    - j_index * sij * self.mu[i] * self.s[i]
                    - cij
                ) / self.s[i] ** 3
                # without parity factor
                self.d0[0, 0, i, j] = cij
                self.d1[0, 0, i, j] = j_index * sij / self.s[i]
                self.d2[0, 0, i, j] = (j_index / self.s[i] ** 3) * (
                    self.mu[i] * sij - j_index * cij * self.s[i]
                )

                # odd parity
                j_index = 2 * j + 1
                cij = np.cos(
                    np.pi * j_index / self.npts * (self.npts + i + 0.5)
                )
                sij = np.sin(
                    np.pi * j_index / self.npts * (self.npts + i + 0.5)
                )

                # without parity factor
                self.d0[1, 0, i, j] = cij
                self.d1[1, 0, i, j] = j_index * sij / self.s[i]
                self.d2[1, 0, i, j] = (j_index / self.s[i] ** 3) * (
                    self.mu[i] * sij - j_index * cij * self.s[i]
                )
                # with parity factor
                self.d0[1, 1, i, j] = cij * self.s[i]
                self.d1[1, 1, i, j] = (
                    j_index * sij - cij * self.mu[i] / self.s[i]
                )
                self.d2[1, 1, i, j] = (
                    -(j_index**2) * cij * self.s[i] ** 2
                    - j_index * sij * self.mu[i] * self.s[i]
                    - cij
                ) / self.s[i] ** 3

    def _prepare_full_expansion(self) -> None:
        """Prepares the coefficient arrays to compute
        the full Chebyshev collocation expansion.
        """
        # multiply derivatives with inverse of T_n (d0) -> obtain unity matrix coeffs0 accompanying term
        for parity in [0, 1]:
            for pf in [0, 1]:
                self.d0[parity, pf, :] = np.linalg.inv(self.d0[parity, pf, :])
                self.d1[parity, pf, :] = np.dot(
                    self.d1[parity, pf, :], self.d0[parity, pf, :]
                )
                self.d2[parity, pf, :] = np.dot(
                    self.d2[parity, pf, :], self.d0[parity, pf, :]
                )

    def solve_eig(
        self,
        coeff0: np.ndarray,
        coeff1: np.ndarray,
        coeff2: np.ndarray,
        parity: int,
        pf: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve 2nd-order differential eigenvalue problem.

        Parameters
        ----------
        coeff0 : np.ndarray
            Zeroth-order coefficient.
        coeff1 : np.ndarray
            First-order coefficient.
        coeff2 : np.ndarray
            Second-order coefficient.
        parity : int
            Parity of the eigenfunctions.
        pf : int
            Parity factor.

        Returns
        -------
        vals : np.ndarray
            Computed eigenvalues.
        vecs : np.ndarray
            Computed eigenvectors.
        """
        # generate the full matrix
        full = (
            np.dot(np.diag(coeff2), self.d2[parity, pf, :])
            + np.dot(np.diag(coeff1), self.d1[parity, pf, :])
            + np.diag(coeff0)
        )
        # compute and return the eigenvalues/-functions
        return np.linalg.eig(full)

    def apply(
        self,
        f: np.ndarray,
        coeff0: np.ndarray,
        coeff1: np.ndarray,
        coeff2: np.ndarray,
        parity: int,
        pf: int,
    ) -> np.ndarray:
        """Apply the differential operator.

        Parameters
        ----------
        f : np.ndarray
            Values of the function to differentiate.
        coeff0 : np.ndarray
            Zeroth-order coefficient.
        coeff1 : np.ndarray
            First-order coefficient.
        coeff2 : np.ndarray
            Second-order coefficient.
        parity : int
            Parity of the function f.
        pf : int
            Parity factor.

        Returns
        -------
        np.ndarray
            Values after applying the operator to f.
        """
        # generate the full matrix
        full = np.dot(np.diag(coeff1), self.d1[parity, pf, :]) + np.diag(coeff0)
        if coeff2 is not None:
            full += np.dot(np.diag(coeff2), self.d2[parity, pf, :])
        # return the result of applying the differential operator
        return np.dot(full, f)

    def diff1(self, f: np.ndarray, parity: int, pf: int) -> np.ndarray:
        """Compute the first derivative.

        Parameters
        ----------
        f : np.ndarray
            Values of the function to differentiate.
        parity : int
            Parity of the function f.
        pf : int
            Parity factor.

        Returns
        -------
        np.ndarray
            Values of the first derivative of f.
        """
        return np.dot(self.d1[parity, pf, :], f)

    def diff2(self, f: np.ndarray, parity: int, pf: int) -> np.ndarray:
        """Compute the second derivative.

        Parameters
        ----------
        f : np.ndarray
            Values of the function to differentiate.
        parity : int
            Parity of the function f.
        pf : int
            Parity factor.

        Returns
        -------
        np.ndarray
            Values of the second derivative of f.
        """
        return np.dot(self.d2[parity, pf, :], f)
