'''Python module containing the expected values for the SuperCouplingCoefficient class of the aesolver.coupling_coefficients module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import typing
import numpy as np
# import custom support module
from expected_values_enumerated_mode_data \
    import expected_vals_mode_independent_names \
        as EVmIn


# store expected attribute values of the
# SuperCouplingCoefficient class
# -- directly after initialization
class AttributesSuperCouplingCoefficientAfterInit:
    """Contains the attributes of the SuperCouplingCoefficient after initialization of the module."""
    
    # attribute declarations
    _kwargs_diff_div: None | dict=None
    _kwargs_diff_terms: None | dict=None
    _diff_terms_method: str='gradient'
    _list_common: list[str, typing.Any]= EVmIn()
    _use_polytropic: bool=False
    _nr_modes: int=3
    _norm_P: np.float64 | None=1.0
    _norm_rho: np.float64 | None=1.0
    _cut_center: bool=True
    
    # retrieve the output attributes in a dictionary
    @classmethod
    def get_dict(cls) -> dict[str, typing.Any]:
        """Retrieves the input arguments in dictionary format."""
        return {
            _k: getattr(cls, _k) for _k in vars(cls)
            if ('__' not in _k) and ('get_dict' != _k)
            }
    