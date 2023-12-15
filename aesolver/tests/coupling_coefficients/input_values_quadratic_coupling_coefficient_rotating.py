'''Python module containing the input values for the QuadraticCouplingCoefficientRotating class of the aesolver.coupling_coefficients module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import modules
import typing
# import support modules
from mock_gyre_detail import GYREDetailDictData as GDdd
from mock_mesa_profile import MESAProfileData as MpD


# store input values for the SuperCouplingCoefficient class
class InputQuadraticCouplingCoefficientRotating:
    """Contains input values for the QuadraticSuperCouplingCoefficient class during tests."""
    
    # define class attributes
    first_dict: dict[str, typing.Any]= \
        GDdd.get_dict(mode_nr=1, nad=False)
    second_dict: dict[str, typing.Any]= \
        GDdd.get_dict(mode_nr=2, nad=False)
    third_dict: dict[str, typing.Any]= \
        GDdd.get_dict(mode_nr=3, nad=False)
    additional_profile_information: dict[str, typing.Any]= \
        MpD.get_dict()
    l_list: list[int]= [2, 2, 2]
    m_list: list[int]= [2, 2, 2]
    use_complex: bool= False
    kwargs_diff_div: dict[str, typing.Any] | None= None
    kwargs_diff_terms: dict[str, typing.Any] | None= None 
    diff_terms_method: str='gradient'
    polytropic: bool=False
    polytrope_data_dict: dict[str, typing.Any] | None= None
    npoints: int= 200
    self_coupling: bool= False
    use_brunt_def: bool= True
    nr_omp_threads: int= 4
    use_symbolic_derivatives: bool= True
    use_parallel: bool=False
    numerical_integration_method: str= 'trapz'
    
    # retrieve the input arguments in a dictionary
    @classmethod
    def get_dict(cls) -> dict[str, typing.Any]:
        """Retrieves the input arguments in dictionary format."""
        return {
            _k: getattr(cls, _k) for _k in vars(cls)
            if ('__' not in _k) and ('get_dict' != _k)
            }
