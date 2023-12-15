'''Creates a mock setup object/class for the ThreeModeStationary test, containing the frequency handler mock object and the solver mock object.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import modules
import numpy as np
# import the two classes that need to be
# put into one supervising class used during tests
from mock_freq_handler_object import \
    MockFreqHandler as MFH
from mock_solver_object_three_mode import \
    MockSolverThreeModeDefault as MStMD


# define the mock setup object/class using default values
class MockSetupObjectDefault:
    """Holds the necessary classes for the testing of 'ThreeModeStationary'."""
    
    # attribute type declarations
    freq_handler: MFH
    solver: MStMD
    
    def __init__(self, mode_frequencies: np.ndarray,
                 driving_rates: np.ndarray, nr_modes: int = 3,
                 azimuthal_wavenumbers: list[int] | None=None,
                 nr_vals: int=5, cc_val: float=2.0,
                 hr_values: list[np.ndarray] | None=None) -> None:
        # initialize and store the frequency handler object
        self.freq_handler = MFH(
            mode_frequencies=mode_frequencies,
            driving_rates=driving_rates
            )
        self.solver = MStMD(
            nr_modes=nr_modes, nr_vals=nr_vals,
            azimuthal_wavenumbers=azimuthal_wavenumbers,
            cc_val=cc_val, hr_values=hr_values
            )
    
    
# define the mock setup object/class using different disc integrals
class MockSetupObject:
    """Holds the necessary classes for the testing of 'ThreeModeStationary'."""
    
    # attribute type declarations
    freq_handler: MFH
    solver: MStMD
    
    def __init__(self, mode_frequencies: np.ndarray,
                 driving_rates: np.ndarray, nr_modes: int = 3,
                 azimuthal_wavenumbers: list[int] | None=None,
                 nr_vals: int=5, cc_val: float=2.0,
                 hr_values: list[np.ndarray] | None=None) -> None:
        # initialize and store the frequency handler object
        self.freq_handler = MFH(
            mode_frequencies=mode_frequencies,
            driving_rates=driving_rates
            )
        self.solver = MStMD(
            nr_modes=nr_modes, nr_vals=nr_vals,
            azimuthal_wavenumbers=azimuthal_wavenumbers,
            cc_val=cc_val, hr_values=hr_values
            )    
        # change the disc integral factors
        self.solver._ek_disc_integrals = \
            2.0 * self.solver._ek_disc_integrals  # now should be [2., 2., 2.] instead of [1., 1., 1.]
    

# define the mock setup object/class using complex disc integrals
class MockSetupObjectComplexDisc:
    """Holds the necessary classes for the testing of 'ThreeModeStationary'."""
    
    # attribute type declarations
    freq_handler: MFH
    solver: MStMD
    
    def __init__(self, mode_frequencies: np.ndarray,
                 driving_rates: np.ndarray, nr_modes: int = 3,
                 azimuthal_wavenumbers: list[int] | None=None,
                 nr_vals: int=5, cc_val: float=2.0,
                 hr_values: list[np.ndarray] | None=None) -> None:
        # initialize and store the frequency handler object
        self.freq_handler = MFH(
            mode_frequencies=mode_frequencies,
            driving_rates=driving_rates
            )
        self.solver = MStMD(
            nr_modes=nr_modes, nr_vals=nr_vals,
            azimuthal_wavenumbers=azimuthal_wavenumbers,
            cc_val=cc_val, hr_values=hr_values
            )    
        # change the disc integral factors
        self.solver._ek_disc_integrals = \
            1.0j * self.solver._ek_disc_integrals  # now should be [1.j, 1.j, 1.j] instead of [1., 1., 1.]
