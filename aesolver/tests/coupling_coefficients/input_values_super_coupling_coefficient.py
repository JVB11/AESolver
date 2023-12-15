'''Python module containing the input values for the SuperCouplingCoefficient class of the aesolver.coupling_coefficients module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import modules
import typing


# store input values for the SuperCouplingCoefficient class
class InputSuperCouplingCoefficient:
    """Contains input values for the SuperCouplingCoefficient class during tests."""
    
    # define class attributes
    nr_modes: int = 3
    kwargs_diff_div: dict[str, typing.Any] | None= None
    kwargs_diff_terms: dict[str, typing.Any] | None= None 
    diff_terms_method: str='gradient'
    polytropic: bool=False
    cut_center: bool=True
    
    # retrieve the input arguments in a dictionary
    @classmethod
    def get_dict(cls) -> dict[str, typing.Any]:
        """Retrieves the input arguments in dictionary format."""
        return {
            _k: getattr(cls, _k) for _k in vars(cls)
            if ('__' not in _k) and ('get_dict' != _k)
            }
    