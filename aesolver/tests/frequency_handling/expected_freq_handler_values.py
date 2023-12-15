'''Python module containing class that contains the expected values for the frequency handler module during tests.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import typing
import numpy as np
import numpy.typing as npt


# construct a generic float array type / complex float array type
my_F = typing.TypeVar("my_F", bound=npt.NBitBase)  # any numpy bit base
f_t = "np.floating[my_F]"  # floating type
cf_t = "np.complexfloating[my_F]"  # complex floating type
call_f_t = typing.Callable[[f_t], f_t]  # floating callable
call_cf_t = typing.Callable[[cf_t], cf_t]  # complex floating callable


# define class of expected values for adiabatic quantities
class ExpectedValuesFreqHandlerAdiabatic:
    """Contains the expected values for adiabatic quantities of the freq_handler object during testing."""
    
    # attribute definitions
    _requested_units: str = 'HZ'
    _n_modes: int = 3
    _angular: bool = False
    _rot_freq: f_t = \
        np.array([0.5], dtype=np.float64)[0] / (2.0 * np.pi)
    _rot_freq_ang: f_t = \
        np.array([0.5], dtype=np.float64)[0]
    _rot_freq_ang_rps: f_t = \
        np.array([0.5], dtype=np.float64)[0]
    _ang_conv_rps: f_t = \
        np.array([1.0], dtype=np.float64)[0]
    _cffs: list[call_f_t | call_cf_t] = [
        lambda x: x, lambda x: x, lambda x: x
        ]
    _ocfr: call_f_t | call_cf_t = \
        lambda x: x.real[0] * 2.0 * np.pi
    _fcfr: call_f_t | call_cf_t = \
        lambda x: x.real[0]
    _ocfc: call_f_t | call_cf_t = \
        lambda x: x[0] * 2.0 * np.pi
    _fcfc: call_f_t | call_cf_t = \
        lambda x: x[0]
    _om_req: bool = False
    # freq: np.ndarray = np.array([2.0, 3.0, 4.0], dtype=np.complex128), in HZ
    _inert_mode_freqs: f_t= \
        np.array([2.0, 3.0, 4.0], dtype=np.float64)
    _inert_mode_omegas: f_t= \
        np.pi * np.array(
            [4.0, 6.0, 8.0], dtype=np.float64
            )
    _corot_mode_freqs: f_t= \
        np.array(
            [2.0, 3.0, 4.0], dtype=np.float64
            ) - (1.0 / (2.0 * np.pi))
    _corot_mode_omegas: f_t= \
        np.pi * np.array(
            [4.0, 6.0, 8.0], dtype=np.float64
            ) - 1.0
    _spin_factors: f_t= \
        1.0 / ((np.pi * np.array(
            [4.0, 6.0, 8.0], dtype=np.float64
            )) - 1.0)
    _dimless_inert_freqs: f_t= \
        np.array([2.0, 3.0, 4.0], dtype=np.float64)
    _dimless_corot_freqs: f_t= \
        np.array(
            [2.0, 3.0, 4.0], dtype=np.float64
            ) - (1.0 / (2.0 * np.pi))
    _dimless_corot_omegas: f_t= \
        np.pi * np.array(
            [4.0, 6.0, 8.0], dtype=np.float64
            ) - 1.0
    _dimless_inert_omegas: f_t= \
        np.pi * np.array(
            [4.0, 6.0, 8.0], dtype=np.float64
            )
    _detuning: f_t= np.array(
        [1.0 - (np.pi * 10.0)], dtype=np.float64
        )

    # retrieve dict of values
    @classmethod
    def get_dict(cls) -> dict[str, typing.Any]:
        """Retrieves the expected value dictionary for the adiabatic attributes of freq_handler.

        Returns
        -------
        dict[str, typing.Any]
            Stores the expected key-value pairs for the adiabatic attributes of freq_handler.
        """
        # return the dict
        return {
            a: getattr(cls, a)
            for a in vars(cls) if ('__' not in a) \
                and ('get_dict' not in a)
        }


# define class of expected values for nonadiabatic quantities
class ExpectedValuesFreqHandlerNonAdiabatic:
    """Contains the expected values for nonadiabatic quantities of the freq_handler object during testing."""
    
    # attribute definitions
    _driving_rates: f_t =\
        np.pi * np.array([0.2, -0.5, -1.0], dtype=np.float64)
    _inert_mode_freqs_nad: f_t= \
        np.array([2.0, 3.0, 4.0], dtype=np.float64)
    _inert_mode_omegas_nad: f_t= \
        np.pi * np.array(
            [4.0, 6.0, 8.0], dtype=np.float64
            )
    _corot_mode_freqs_nad: f_t= \
        np.array(
            [2.0, 3.0, 4.0], dtype=np.float64
            ) - (1.0 / (2.0 * np.pi))
    _corot_mode_omegas_nad: f_t= \
        np.pi * np.array(
            [4.0, 6.0, 8.0], dtype=np.float64
            ) - 1.0
    _quality_factors: f_t = \
        np.array([20.0, -12.0, -8.0], dtype=np.float64) \
            + (np.array([-5.0, 2.0, 1.0], dtype=np.float64) / np.pi)
    _quality_factors_nad: f_t = \
        np.array([20.0, -12.0, -8.0], dtype=np.float64)\
            + (np.array([-5.0, 2.0, 1.0], dtype=np.float64) / np.pi)
    _quality_factor_products: f_t = \
        np.array([-240.0, -160.0, 96.0], dtype=np.float64) \
            + (np.array([100.0, 60.0, -28.0], dtype=np.float64) / np.pi) \
                + (np.array([-10.0, -5.0, 2.0], dtype=np.float64) / (np.pi**2.0))
    _quality_factor_products_nad: f_t = \
        np.array([-240.0, -160.0, 96.0], dtype=np.float64) \
            + (np.array([100.0, 60.0, -28.0], dtype=np.float64) / np.pi) \
                + (np.array([-10.0, -5.0, 2.0], dtype=np.float64) / (np.pi**2.0))
    _gamma_sum: f_t = \
        np.array([-1.3 * np.pi], dtype=np.float64)
    _dimless_inert_freqs_nad: f_t= \
        np.array([2.0, 3.0, 4.0], dtype=np.float64)
    _dimless_inert_omegas_nad: f_t= \
        np.pi * np.array(
            [4.0, 6.0, 8.0], dtype=np.float64
            )
    _dimless_corot_freqs_nad: f_t= \
        np.array(
            [2.0, 3.0, 4.0], dtype=np.float64
            ) - (1.0 / (2.0 * np.pi))
    _dimless_corot_omegas_nad: f_t= \
        np.pi * np.array(
            [4.0, 6.0, 8.0], dtype=np.float64
            ) - 1.0 
    _detuning_nad: f_t= np.array(
        [1.0 - (np.pi * 10.0)], dtype=np.float64
        )
    _q_fac_nad: f_t = np.array(
        [(1.0 - (10.0 * np.pi)) / (-1.3 * np.pi)], dtype=np.float64
        )
    _q1_fac_nad: f_t = np.array(
        [(5.0 / np.pi) - 50.0], dtype=np.float64
        )    
    _q_fac: f_t = np.array(
        [(1.0 - (10.0 * np.pi)) / (-1.3 * np.pi)], dtype=np.float64
        )
    _q1_fac: f_t = np.array(
        [(5.0 / np.pi) - 50.0], dtype=np.float64
        )

    # retrieve dict of values
    @classmethod
    def get_dict(cls) -> dict[str, typing.Any]:
        """Retrieves the expected value dictionary for the adiabatic attributes of freq_handler.

        Returns
        -------
        dict[str, typing.Any]
            Stores the expected key-value pairs for the adiabatic attributes of freq_handler.
        """
        # return the dict
        return {
            a: getattr(cls, a)
            for a in vars(cls) if ('__' not in a) \
                and ('get_dict' not in a)
        }
