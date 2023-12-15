'''Python module containing subclass used to perform quadratic grid computations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import logging
import typing
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# relative imports
# - used to load the superclass
from .solver_base_classes import QuadraticAESolver

# - used to load mode combination data
from ..mode_input_generator import InputGen

# - computes stationary solutions for A = B + C
from ..stationary_solving import ThreeModeStationary

# type checking imports
if typing.TYPE_CHECKING:
    from typing import Any
    
    
# set up logger
logger = logging.getLogger(__name__)


# use this adjusted fontset for every figure (obtain correct epsilon)
plt.rcParams['mathtext.fontset'] = 'cm'
        
        
class QuadraticAEGridSolver(QuadraticAESolver):
    """The class that handles grid computations for the solving of the quadratic amplitude equations in rapidly rotating gravity-mode pulsating stars.
    
    Parameters
    ----------
    
    """
    
    # attribute type declarations
    
    def __init__(self, sys_args: list, called_from_pytest: bool = False, pytest_path: Path | None = None, rot_pct: int = 20, use_complex: bool = False, adiabatic: bool = True, use_rotating_formalism: bool = True, base_dir: str | None = None, alternative_directory_path: str | None = None, alternative_file_name: str | None = None, use_symbolic_profiles: bool = True, use_brunt_def: bool = True, mode_selection_dict: 'dict[str, Any] | None' = None, nr_omp_threads: int = 4, use_parallel: bool = False, ld_function: str = 'eddington', numerical_integration_method: str = 'trapz', use_cheby_integration: bool = False, cheby_order_multiplier: int = 4, cheby_blow_up_protection: bool = False, cheby_blow_up_factor: float = 1000, stationary_solving: bool = True, ad_path: str | None = None, nad_path: str | None = None, inlist_path: str | None = None, toml_use: bool = False, get_debug_info: bool = False, terms_cc_conj: tuple[bool, bool, bool] | bool | None = None, polytrope_comp: bool = False, analytic_polytrope: bool = False, polytrope_mass: float = 3, polytrope_radius: float = 4.5, progress_bars: bool=False) -> None:
        # initialize the super-class
        super().__init__(sys_args, called_from_pytest, pytest_path, rot_pct, use_complex, adiabatic, use_rotating_formalism, base_dir, alternative_directory_path, alternative_file_name, use_symbolic_profiles, use_brunt_def, nr_omp_threads, use_parallel, ld_function, numerical_integration_method, use_cheby_integration, cheby_order_multiplier, cheby_blow_up_protection, cheby_blow_up_factor, stationary_solving, ad_path, nad_path, inlist_path, toml_use, get_debug_info, terms_cc_conj, polytrope_comp, analytic_polytrope, polytrope_mass, polytrope_radius, progress_bars)
        # set up the mode information object
        if mode_selection_dict is None:
                self._mode_info_object = InputGen(
                    damped_n_low1=30,
                    damped_n_high1=45,
                    driven_n_low=15,
                    driven_n_high=25,
                    mode_l=[2, 1, 1],
                    mode_m=[2, 1, 1],
                    damped_n_low2=None,
                    damped_n_high2=None,
                )
        else:
            self._mode_info_object = InputGen(**mode_selection_dict)
        # initialize the lists that will hold the necessary information
        # - mode coupling information
        self._coupling_info = [
            tuple() for _ in self._mode_info_object.combinations_radial_orders
        ]
        # - mode statistics information
        self._coupling_stat_info = [tuple() for _ in self._coupling_info]
        # - stationary solution information
        self._stat_sol_info = [dict() for _ in self._coupling_info]
        # - mode frequency information
        self._mode_freq_info = [tuple() for _ in self._coupling_info]
        # - hyperbolicity (of fixed point) information
        self._hyperbolicity_info = [False for _ in self._coupling_info]
        
    # make the class a callable so that the requested data can be computed
    def __call__(
        self,
    ) -> None:
        # compute the mode coupling coefficient for the different mode triads and store the information in this solver object, as well as the stationary solutions
        # - CHECK if you want to display the progress bar for the computations
        if self._display_progress_bars:
            for _i in tqdm(
                range(len(self._coupling_info)), desc='Grid computations'
            ):
                self._call_grid_computations(info_index=_i)
        else:
            for _i in range(len(self._coupling_info)):
                self._call_grid_computations(info_index=_i)
        # store the relevant mode statistics
        self._store_relevant_statistics()
        
    def _call_grid_computations(self, info_index: int) -> None:
        """Performs the necessary computations for the grid.

        Parameters
        ----------
        info_index : int
            Index that selects the information for which the computations are performed (i.e. the combination number).
        """
        # compute the coupling for a single mode triad
        precheck = self.compute_coupling_one_triad(
            radial_order_combo_nr=info_index
        )
        # compute the stationary solutions, if necessary and store the theoretical amplitudes and observable fluctuations, when not using polytropic information
        if not self._polytrope_comp:
            self._stat_sol_info[info_index] = ThreeModeStationary(
                precheck,
                self._freq_handler,
                self._hyperbolicity_info[info_index],
                solver_object=self,
            )()
        else:
            self._stat_sol_info[info_index] = None
            
    # store the relevant information of the specific mode triad for the mode coupling coefficient statistics plot
    def _store_relevant_statistics(self) -> None:
        """Method that stores the relevant statistics for the mode triad that is currently loaded, so that a statistics plot can be generated."""
        # retrieve and store the relevant mode statistics
        for _j, (_avg_rad, _wu) in enumerate(
            zip(
                self._mode_info_object.average_radial_orders,
                self._mode_info_object.wu_difference_statistics,
            )
        ):
            # store the mode statistics
            self._coupling_stat_info[_j] = (_avg_rad, _wu)
         
    # generate loading kwargs
    def _generate_loading_kwargs(self, radial_order_combo_nr: int=0) -> 'dict[str, Any]':
        return {**self._info_dict_info_dirs,
            'infer_dir': False,
            'search_l': self._mode_info_object.mode_l,
            'search_m': self._mode_info_object.mode_m,
            'search_n': self._mode_info_object.combinations_radial_orders[
                radial_order_combo_nr
            ],}
