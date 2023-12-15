'''Python module containing enumeration objects that hold the expected values for the ThreeModeStationary call outcome in aesolver.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import typing
import numpy as np
# import pytest util class
from pytest_util_classes import EVs


# subclass the pytest util class to gain additional dictionary functionality
class EVExtend(EVs):
    """Extends the EVs class with a method that allows one to replace underscores in keys in the output dictionary with spaces."""
    
    @classmethod
    def get_replaced_dict(cls) -> dict[str, typing.Any]:
        """Defines an output dictionary in which the keys have underscores replaced with spaces.

        Returns
        -------
        dict[str, typing.Any]
            The modified output dictionary.
        """
        # obtain the enumeration dictionary
        my_enum_dict = cls.get_enumeration_dict()
        # replace underscores in keys with spaces
        return {
            _k.replace('_', ' '): _v
            for _k, _v in my_enum_dict.items()
            }


# test (default) output for precheck NOT satisfied
class DefaultOutputPreCheckNotOk(EVExtend):
    """Stores the default output when the pre-check is not OK for our test runs."""
    
    HYPERBOLIC: bool = True
    LINEARLY_STABLE: bool = False
    STATIONARY_RELATIVE_LUMINOSITY_PHASE: \
        np.float64 | None = None
    STATIONARY_RELATIVE_PHASE: \
        np.float64 | None = None
    CRITICAL_PARENT_AMPLITUDE: \
        np.float64 | None = None
    CRITICAL_Q_FACTOR: np.float64 = np.array(
        [2.5], dtype=np.float64
        )[0]
    Q_FACTOR: np.float64 = np.array(
        [5.0], dtype=np.float64
        )[0]
    DETUNINGS: np.float64 = np.array(
        [-5.0], dtype=np.float64
        )[0]  # TODO: rename to 'detuning' in actual code
    GAMMA: np.float64 = np.array(
        [-1.0], dtype=np.float64
        )[0]
    LINEAR_DRIVING_RATES: np.ndarray = \
        np.array([1.0, -1.0, -1.0],
                 dtype=np.float64).tobytes()
    SURFACE_FLUX_VARIATIONS: np.ndarray = \
        np.zeros((3,), dtype=np.float64).tobytes()
    EQUILIBRIUM_AMPLITUDES: np.ndarray = \
        np.zeros((3,), dtype=np.float64).tobytes()


# test (default) output for precheck satisfied
class DefaultOutputPreCheckOk(EVExtend):
    """Stores the default output when the pre-check is OK for our test runs."""
    
    HYPERBOLIC: bool = True
    LINEARLY_STABLE: bool = True
    STATIONARY_RELATIVE_LUMINOSITY_PHASE: \
        np.float64 | None = np.arctan(np.array(
            [-1.0 / 5.0], dtype=np.float64
            ))[0]
    # The stationary relative luminosity phase is computed as:
    #   Azimuthal factors:
    #       az_factors = ZEROS because the azimuthal angle defaulted to
    #                    zero when az_ang_periastron == None (| 0.0)
    #       _az_sum = az_factors[0] - sum(az_factors[1:]) = 0.0
    #   Relative luminosity phase:
    #       angles = COMPLEX_ANGLE(disc_multi_factors)
    #              = ZEROS because disc_multi_factors = ZEROS
    #       rel_lum_phase = angles[0] - sum(angles[1:]) = 0.0
    #   Relative stationary luminosity phase:
    #       rel_stat_lum_phase = rel_stat_phase + rel_lum_phase
    #                            + _az_sum = ARCTAN(-1.0 / 5.0)
    STATIONARY_RELATIVE_PHASE: \
        np.float64 | None = np.arctan(np.array(
            [-1.0 / 5.0], dtype=np.float64
            ))[0]
    # The stationary relative phase is computed as:
    #   rel_stat_phase = ARCTAN(-1.0 / 5.0) = -0.19739555984988078
    CRITICAL_PARENT_AMPLITUDE: \
        np.float64 | None = (0.5 * np.sqrt(np.array(
            [7.25 / 48.0], dtype=np.float64
            )))[0]
    # Critical parent amplitude is computed as:
    #   pre_fac_crit = (1.0 + (2.5)^2) * (1.0 / (4.0 * 12.0)) = 7.25 / 48.0
    #   a_crit = SQRT(pre_fac_crit) / 2.0 = 0.5*SQRT(7.25/48.0) = 0.1943203969393503
    CRITICAL_Q_FACTOR: np.float64 = np.array(
        [2.5], dtype=np.float64
        )[0]
    Q_FACTOR: np.float64 = np.array(
        [5.0], dtype=np.float64
        )[0]
    DETUNINGS: np.float64 = np.array(
        [-5.0], dtype=np.float64
        )[0]  # TODO: rename to 'detuning' in actual code
    GAMMA: np.float64 = np.array(
        [-1.0], dtype=np.float64
        )[0]
    LINEAR_DRIVING_RATES: np.ndarray = \
        np.array([1.0, -1.0, -1.0],
                 dtype=np.float64).tobytes()
    SURFACE_FLUX_VARIATIONS: np.ndarray = \
        np.zeros((3,), dtype=np.float64).tobytes()
    # This is ZERO, because the disc multiplication factors are ZERO
    EQUILIBRIUM_AMPLITUDES: np.ndarray = \
        (0.5 * np.sqrt(np.array(
            [13.0 / 24.0, 13.0 / 16.0, 13.0 / 12.0],
            dtype=np.float64
            ))).tobytes()
    # Theoretical amplitudes are computed as:
    #   pre_facs = 1.0 / (4.0 * [-6.0, -8.0, 12.0])
    #   pre_facs_q = (1.0 + 25.0) * pre_facs = [-26.0/24.0, -26.0/32.0, 26.0/48.0] = [-13/12, -13/16, 13/24]
    #   equilibrium_amplitudes:
    #       a1 = SQRT(pre_facs_q[2]) / 2.0 = 0.5*[SQRT(13/24)] = 0.3679900360969936
    #       a2 = SQRT(-pre_facs_q[1]) / 2.0 = 0.5*[SQRT(13/16)] = 0.45069390943299864
    #       a3 = SQRT(-pre_facs_q[0]) / 2.0 = 0.5*[SQRT(13/12)] = 0.5204164998665332
   
 
# test (default) output for precheck satisfied and disc integral adapted
class DefaultOutputPreCheckOkDisc(EVExtend):
    """Stores the default output when the pre-check is OK for our test runs and the disc integral factors are changed."""
    
    HYPERBOLIC: bool = True
    LINEARLY_STABLE: bool = True
    STATIONARY_RELATIVE_LUMINOSITY_PHASE: \
        np.float64 | None = np.arctan(np.array(
            [-1.0 / 5.0], dtype=np.float64
            ))[0]
    # The stationary relative luminosity phase is computed as:
    #   Azimuthal factors:
    #       az_factors = ZEROS because the azimuthal angle defaulted to
    #                    zero when az_ang_periastron == None (| 0.0)
    #       _az_sum = az_factors[0] - sum(az_factors[1:]) = 0.0
    #   Relative luminosity phase:
    #       angles = COMPLEX_ANGLE(disc_multi_factors)
    #              = ZEROS because disc_multi_factors = ZEROS
    #       rel_lum_phase = angles[0] - sum(angles[1:]) = 0.0
    #   Relative stationary luminosity phase:
    #       rel_stat_lum_phase = rel_stat_phase + rel_lum_phase
    #                            + _az_sum = ARCTAN(-1.0 / 5.0)
    STATIONARY_RELATIVE_PHASE: \
        np.float64 | None = np.arctan(np.array(
            [-1.0 / 5.0], dtype=np.float64
            ))[0]
    # The stationary relative phase is computed as:
    #   rel_stat_phase = ARCTAN(-1.0 / 5.0) = -0.19739555984988078
    CRITICAL_PARENT_AMPLITUDE: \
        np.float64 | None = (0.5 * np.sqrt(np.array(
            [7.25 / 48.0], dtype=np.float64
            )))[0]
    # Critical parent amplitude is computed as:
    #   pre_fac_crit = (1.0 + (2.5)^2) * (1.0 / (4.0 * 12.0)) = 7.25 / 48.0
    #   a_crit = SQRT(pre_fac_crit) / 2.0 = 0.5*SQRT(7.25/48.0) = 0.1943203969393503
    CRITICAL_Q_FACTOR: np.float64 = np.array(
        [2.5], dtype=np.float64
        )[0]
    Q_FACTOR: np.float64 = np.array(
        [5.0], dtype=np.float64
        )[0]
    DETUNINGS: np.float64 = np.array(
        [-5.0], dtype=np.float64
        )[0]  # TODO: rename to 'detuning' in actual code
    GAMMA: np.float64 = np.array(
        [-1.0], dtype=np.float64
        )[0]
    LINEAR_DRIVING_RATES: np.ndarray = \
        np.array([1.0, -1.0, -1.0],
                 dtype=np.float64).tobytes()
    SURFACE_FLUX_VARIATIONS: np.ndarray = \
        (0.5 * np.sqrt(np.array(
            [13.0 / 24.0, 13.0 / 16.0, 13.0 / 12.0],
            dtype=np.float64
            ))).tobytes()
    # This is not ZERO, because the disc multiplication factors are not ZERO!
    # Computation of the surface flux variations:
    #   disc multiplication factors:
    #       disc_multi_factors = np.abs(_tk * surf_lums - _ek * surf_xirs)
    #                          = |[1.+ 0.j, 1.+ 0.j, 1.+ 0.j] * [1.+ 0.j, 1.+ 0.j, 1.+ 0.j] - [2.+ 0.j, 2.+ 0.j, 2.+ 0.j] * [1.+ 0.j, 1.+ 0.j, 1.+ 0.j]|
    #                          = [1., 1., 1.]
    #   luminosity fluctuations:
    #       _f_arr = [a1 * disc_multi_factors[0], a2 * disc_multi_factors[1], a3 * disc_multi_factors[2]]
    #              = [0.5*[SQRT(13/24)], 0.5*[SQRT(13/16)], 0.5*[SQRT(13/12)]] --> see below!
    #   hough functions for the observer:
    #       houghs_observer = [1.0, 1.0, 1.0] --> mu_val_spin_inclination == None!
    #   surface flux variations:
    #       flucs = _f_arr * houghs_observer = [0.5*[SQRT(13/24)], 0.5*[SQRT(13/16)], 0.5*[SQRT(13/12)]]
    EQUILIBRIUM_AMPLITUDES: np.ndarray = \
        (0.5 * np.sqrt(np.array(
            [13.0 / 24.0, 13.0 / 16.0, 13.0 / 12.0],
            dtype=np.float64
            ))).tobytes()
    # Theoretical amplitudes are computed as:
    #   pre_facs = 1.0 / (4.0 * [-6.0, -8.0, 12.0])
    #   pre_facs_q = (1.0 + 25.0) * pre_facs = [-26.0/24.0, -26.0/32.0, 26.0/48.0] = [-13/12, -13/16, 13/24]
    #   equilibrium_amplitudes:
    #       a1 = SQRT(pre_facs_q[2]) / 2.0 = 0.5*[SQRT(13/24)] = 0.3679900360969936
    #       a2 = SQRT(-pre_facs_q[1]) / 2.0 = 0.5*[SQRT(13/16)] = 0.45069390943299864
    #       a3 = SQRT(-pre_facs_q[0]) / 2.0 = 0.5*[SQRT(13/12)] = 0.5204164998665332


# test (default) output for precheck satisfied and disc integral adapted
class DefaultOutputPreCheckOkDiscComplex(EVExtend):
    """Stores the default output when the pre-check is OK for our test runs and the disc integral factors are changed to complex values."""
    
    HYPERBOLIC: bool = True
    LINEARLY_STABLE: bool = True
    STATIONARY_RELATIVE_LUMINOSITY_PHASE: \
        np.float64 | None = np.arctan(np.array(
            [-1.0 / 5.0], dtype=np.float64
            ))[0]
    # The stationary relative luminosity phase is computed as:
    #   Azimuthal factors:
    #       az_factors = ZEROS because the azimuthal angle defaulted to
    #                    zero when az_ang_periastron == None (| 0.0)
    #       _az_sum = az_factors[0] - sum(az_factors[1:]) = 0.0
    #   Relative luminosity phase:
    #       angles = COMPLEX_ANGLE(disc_multi_factors)
    #              = ZEROS because disc_multi_factors = ZEROS
    #       rel_lum_phase = angles[0] - sum(angles[1:]) = 0.0
    #   Relative stationary luminosity phase:
    #       rel_stat_lum_phase = rel_stat_phase + rel_lum_phase
    #                            + _az_sum = ARCTAN(-1.0 / 5.0)
    STATIONARY_RELATIVE_PHASE: \
        np.float64 | None = np.arctan(np.array(
            [-1.0 / 5.0], dtype=np.float64
            ))[0]
    # The stationary relative phase is computed as:
    #   rel_stat_phase = ARCTAN(-1.0 / 5.0) = -0.19739555984988078
    CRITICAL_PARENT_AMPLITUDE: \
        np.float64 | None = (0.5 * np.sqrt(np.array(
            [7.25 / 48.0], dtype=np.float64
            )))[0]
    # Critical parent amplitude is computed as:
    #   pre_fac_crit = (1.0 + (2.5)^2) * (1.0 / (4.0 * 12.0)) = 7.25 / 48.0
    #   a_crit = SQRT(pre_fac_crit) / 2.0 = 0.5*SQRT(7.25/48.0) = 0.1943203969393503
    CRITICAL_Q_FACTOR: np.float64 = np.array(
        [2.5], dtype=np.float64
        )[0]
    Q_FACTOR: np.float64 = np.array(
        [5.0], dtype=np.float64
        )[0]
    DETUNINGS: np.float64 = np.array(
        [-5.0], dtype=np.float64
        )[0]  # TODO: rename to 'detuning' in actual code
    GAMMA: np.float64 = np.array(
        [-1.0], dtype=np.float64
        )[0]
    LINEAR_DRIVING_RATES: np.ndarray = \
        np.array([1.0, -1.0, -1.0],
                 dtype=np.float64).tobytes()
    SURFACE_FLUX_VARIATIONS: np.ndarray = \
        np.sqrt(np.array(
            [13.0 / 48.0, 13.0 / 32.0, 13.0 / 24.0],
            dtype=np.float64
            )).tobytes()
    # This is not ZERO, because the disc multiplication factors are not ZERO!
    # Computation of the surface flux variations:
    #   disc multiplication factors:
    #       disc_multi_factors = np.abs(_tk * surf_lums - _ek * surf_xirs)
    #                          = |[1.+ 0.j, 1.+ 0.j, 1.+ 0.j] * [1.+ 0.j, 1.+ 0.j, 1.+ 0.j] - [0.+ 1.j, 0.+ 1.j, 0.+ 1.j] * [1.+ 0.j, 1.+ 0.j, 1.+ 0.j]|
    #                          = [SQRT(2), SQRT(2), SQRT(2)]
    #   luminosity fluctuations:
    #       _f_arr = [a1 * disc_multi_factors[0], a2 * disc_multi_factors[1], a3 * disc_multi_factors[2]]
    #              = [[SQRT(13/48)], [SQRT(13/32)], [SQRT(13/24)]] --> see below for amplitudes!
    #   hough functions for the observer:
    #       houghs_observer = [1.0, 1.0, 1.0] --> mu_val_spin_inclination == None!
    #   surface flux variations:
    #       flucs = _f_arr * houghs_observer = [[SQRT(13/48)], [SQRT(13/32)], [SQRT(13/24)]]
    EQUILIBRIUM_AMPLITUDES: np.ndarray = \
        (0.5 * np.sqrt(np.array(
            [13.0 / 24.0, 13.0 / 16.0, 13.0 / 12.0],
            dtype=np.float64
            ))).tobytes()
    # Theoretical amplitudes are computed as:
    #   pre_facs = 1.0 / (4.0 * [-6.0, -8.0, 12.0])
    #   pre_facs_q = (1.0 + 25.0) * pre_facs = [-26.0/24.0, -26.0/32.0, 26.0/48.0] = [-13/12, -13/16, 13/24]
    #   equilibrium_amplitudes:
    #       a1 = SQRT(pre_facs_q[2]) / 2.0 = 0.5*[SQRT(13/24)] = 0.3679900360969936
    #       a2 = SQRT(-pre_facs_q[1]) / 2.0 = 0.5*[SQRT(13/16)] = 0.45069390943299864
    #       a3 = SQRT(-pre_facs_q[0]) / 2.0 = 0.5*[SQRT(13/12)] = 0.5204164998665332
