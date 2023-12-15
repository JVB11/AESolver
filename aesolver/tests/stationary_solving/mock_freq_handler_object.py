'''Python module containing the mock object
that mimics a frequency_handler object.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import numpy as np


# define the mock frequency handler class
class MockFreqHandler:
    """Mock frequency handler object to be used during testing of the aesolver.pre_checks module."""
    
    # attribute type declarations
    mode_freqs: np.ndarray
    driving_rates: np.ndarray
    gamma_sum: float
    quality_factors: np.ndarray
    quality_factor_products: np.ndarray
    _detuning: float
    
    def __init__(self, mode_frequencies: np.ndarray,
                 driving_rates: np.ndarray) -> None:
        # store the mock data
        self.mode_freqs = mode_frequencies
        self.driving_rates = driving_rates
        self._compute_gamma_sum()
        self._compute_quality_factors()
        self._compute_quality_factor_products()
        self._compute_detuning()

    # get the detunings
    def get_detunings(self) -> np.ndarray:
        """Compute the (ANGULAR FREQUENCY) detunings for a nonlinear coupling. TODO: should remove this function from the library since the detuning should be the same in all frames!
        
        Returns
        -------
        np.ndarray
            Contains the detunings in the co-rotating and inertial frame. First element = co-rotating frame, second = inertial.
        """
        return np.array(
            [self._detuning, self._detuning], dtype=np.float64
            ).reshape((-1, 1))

    def _compute_detuning(self) -> None:
        """Computes the detuning."""
        self._detuning = self.mode_freqs[0] \
            - self.mode_freqs[1] - self.mode_freqs[2]

    def _compute_gamma_sum(self) -> None:
        """Computes the summed driving/damping rates of the mode triad."""
        self.gamma_sum = self.driving_rates.sum()

    def _compute_quality_factors(self) -> None:
        """Computes the quality factors of the mode triad."""
        # since the mode frequencies are: [2.0, 3.0, 4.0]
        # and the driving rates are: [1.0, -1.0, -1.0]
        # the quality factors should be: [2.0, -3.0, -4.0]
        self.quality_factors = \
            self.mode_freqs / self.driving_rates
            
    def _compute_quality_factor_products(self) -> None:
        """Compute the quality factor products of the mode triad."""
        # init
        self.quality_factor_products = \
            np.zeros((3,), dtype=np.float64)
        # fill: since the quality factors are: [2.0, -3.0, -4.0]
        self.quality_factor_products[0] = \
            self.quality_factors[0] * \
                self.quality_factors[1]  # should be -6.0
        self.quality_factor_products[1] = \
            self.quality_factors[0] * \
                self.quality_factors[2]  # should be -8.0
        self.quality_factor_products[2] = \
            self.quality_factors[2] * \
                self.quality_factors[1]  # should be 12.0
