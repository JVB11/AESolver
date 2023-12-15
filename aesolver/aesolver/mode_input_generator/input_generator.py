"""Python module containing class that generates input for the comparison of adiabatic and non-adiabatic computations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import statistics
import numpy as np

# from custom modules import combination computation objects
from py_combinations import SingleSetCombos, SymmetricProductCombos


# set up the logger
logger = logging.getLogger(__name__)


# class that computes necessary input for comparison of
# non-adiabatic and adiabatic results
class AdNadModeInfoInputGenerator:
    """Python class containing method(s) that generate the necessary mode information for the comparison of non-adiabatic and adiabatic GYRE pulsation computation results.

    Parameters
    ----------
    lower_n_low : int
        The lower bound for the radial orders of the low-l modes.
    lower_n_high : int
        The upper bound for the radial orders of the low-l modes.
    higher_n_low : int
        The lower bound for the radial orders of the high-l modes.
    higher_n_high : int
        The upper bound for the radial orders of the high-l modes.
    mode_l : list[int]
        Contains the spherical mode degrees.
    mode_m : list[int]
        Contains the azimuthal mode orders.
    """

    mode_l: list
    mode_m: list
    _driven_rad_order_range: range
    _damped_rad_order_range_1: range
    _damped_rad_order_range_2: range | None
    combinations_radial_orders: list
    average_radial_orders: list
    wu_difference_statistics: list

    def __init__(
        self,
        mode_l: list[int],
        mode_m: list[int],
        driven_n_low: int,
        driven_n_high: int,
        damped_n_low1: int,
        damped_n_high1: int,
        damped_n_low2: int | None = None,
        damped_n_high2: int | None = None,
    ) -> None:
        # store the information on mode nonradial geometry
        self.mode_l = mode_l
        self.mode_m = mode_m
        # store the information on the ranges of radial orders
        self._driven_rad_order_range = range(driven_n_low, driven_n_high + 1)
        self._damped_rad_order_range_1 = range(
            damped_n_low1, damped_n_high1 + 1
        )
        if (damped_n_low2 is not None) and (damped_n_high2 is not None):
            self._damped_rad_order_range_2 = range(
                damped_n_low2, damped_n_high2 + 1
            )
        else:
            self._damped_rad_order_range_2 = None
        # generate the input combinations
        logger.debug('Generating input combinations.')
        self._generate_input_combinations()
        logger.debug('Input combinations generated.')
        # generate the mode statistics
        logger.debug('Computing mode statistics.')
        self._compute_average_radial_orders()
        self._compute_wu_difference_statistics()
        logger.debug('Mode statistics computed.')

    def _get_damped_combinations(
        self, include_diagonal: bool = True
    ) -> list[tuple]:
        """Retrieve and store the combinations of radial orders for the damped modes to be loaded. These potentially include both harmonic frequencies and combination frequencies.

        Returns
        -------
        list[tuple]
            Contains tuples of radial orders of second-order combinations of damped modes.
        """
        # initialize the object
        my_object = SymmetricProductCombos(
            A=self._damped_rad_order_range_1,
            B=self._damped_rad_order_range_2,
            include_diagonal=include_diagonal,
        )
        # return the combination list
        return my_object.l

    def _get_damped_combinations_single_range(self) -> list[tuple]:
        """Utility method that generates second-order combinations of radial orders for the damped modes in a parametric resonance.

        Returns
        -------
        list[tuple]
            Contains tuples of radial orders of second-order combinations of damped modes.
        """
        # initialize the object
        my_object = SingleSetCombos(my_set=self._damped_rad_order_range_1)
        # return the combination list
        return my_object.l

    def _generate_input_combinations(self) -> None:
        """Generates the radial order input combinations for nad-ad comparison."""
        # check if the second damped mode range is None
        if self._damped_rad_order_range_2 is None:
            # generate combinations of frequencies where the radial orders of the damped modes are chosen from one range of values
            self.combinations_radial_orders = [
                [_higher_rad_ord, *_lower_rad_ord_combo]
                for _higher_rad_ord in self._driven_rad_order_range
                for _lower_rad_ord_combo in self._get_damped_combinations_single_range()
            ]
            # no harmonics present for this type of combination seeking
            self.harmonics_radial_orders = []
        else:
            # generate combinations of frequencies where the radial orders of the two damped modes are chosen from two (individual) ranges of values
            self.combinations_radial_orders = [
                [_driven_rad_ord, *_damped_rad_ord_combo]
                for _driven_rad_ord in self._driven_rad_order_range
                for _damped_rad_ord_combo in self._get_damped_combinations()
            ]

    def _compute_average_radial_orders(self) -> None:
        """Computes and stores the average radial orders for the mode triads."""
        self.average_radial_orders = [
            statistics.mean(_rad_ord_combo)
            for _rad_ord_combo in self.combinations_radial_orders
        ]

    def _compute_wu_difference_statistics(self) -> None:
        """Computes and stores the Wu & Goldreich (2001) and Lee(2012) radial order (difference) statistics."""
        self.wu_difference_statistics = [
            _rad_ord_combo[0] - np.abs(_rad_ord_combo[1] - _rad_ord_combo[2])
            for _rad_ord_combo in self.combinations_radial_orders
        ]
