'''Contains typing classes used in the solver run module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
from typing import TypedDict


class ModeInfoOrPolyDict(TypedDict):
    rot_pct: int
    base_dir: str
    alternative_directory_path: str
    alternative_file_name: str
    verbose: bool
    debug: bool


class ModeSelectionDict(TypedDict):
    driven_n_low: int
    driven_n_high: int
    damped_n_low1: int
    damped_n_high1: int
    damped_n_low2: int
    damped_n_high2: int
    mode_l: list[int]
    mode_m: list[int]


class ModeComputationInfoDict(TypedDict):
    adiabatic: bool
    use_complex: bool
    use_symbolic_profiles: bool
    use_brunt_def: bool
    use_rotating_formalism: bool
    terms_cc_conj: bool
    polytrope_comp: bool
    analytic_polytrope: bool
 
    
class ComputationDict(TypedDict):
    use_parallel: bool
    compute_grid: bool
    compute_profile: bool
    progress_bars: bool
    
    
class SavingDict(TypedDict):
    base_dir: str
    save_dir: str
    extra_specifier: str
    initial_save_format: str | list[str]
    
    
class PolyModelDict(TypedDict):
    polytrope_mass: float
    polytrope_radius: float
    
    
class AngularIntegrationDict(TypedDict):
    numerical_integration_method: str
    use_cheby_integration: bool
    cheby_order_multiplier: float
    cheby_blow_up_protection: bool
    cheby_blow_up_factor: float
    
    
class CouplingCoefficientProfileSelectionDict(TypedDict):
    paren_nt_n: int
    daughter_1_n: int
    daughter_2_j: int
    mode_l: tuple[int, int, int]
    mode_m: tuple[int, int, int]
    
    
class CouplingCoefficientProfilePlotDict(TypedDict):
    view_kernels: bool
    view_term_by_term: bool
