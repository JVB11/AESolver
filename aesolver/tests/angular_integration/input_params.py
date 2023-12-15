'''Python module containing two enumeration classes that define the input parameters and their variables for our test.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import json
# import pytest_util_classes
from pytest_util_classes import EVs, EPPs


# define the input parameters
class InputParameters(EPPs):
    """Defines the input parameters for our tests."""
    
    TWO_DATA = None
    TWO_DATA_DIV_SIN = None
    TWO_DATA_MU = None
    TWO_DATA_PARALLEL = None
    TWO_DATA_CONJ = None
    TWO_DATA_CHECK_M = None
    TWO_DATA_LEN = "length of m_list is wrong"
    THREE_DATA = None
    THREE_DATA_DIV_SIN = None
    THREE_DATA_MU = None
    THREE_DATA_PARALLEL = None
    THREE_DATA_CONJ = None
    THREE_DATA_CHECK_M = None
    THREE_DATA_LEN = "length of m_list is wrong"


# define the input parameters
class InputParametersNoAlternativeDivSinProcedure(EPPs):
    """Defines the input parameters for our tests when we do not make use of alternative div_sin procedures(!)."""
    
    TWO_DATA = None
    TWO_DATA_DIV_SIN = "No alternative div_sin procedure used: larger value obtained!"
    TWO_DATA_MU = None
    TWO_DATA_PARALLEL = None
    TWO_DATA_CONJ = None
    TWO_DATA_CHECK_M = None
    TWO_DATA_LEN = "length of m_list_for_azimuthal_check is wrong"
    THREE_DATA = None
    THREE_DATA_DIV_SIN = "No alternative div_sin procedure used: larger value obtained!"
    THREE_DATA_MU = None
    THREE_DATA_PARALLEL = None
    THREE_DATA_CONJ = None
    THREE_DATA_CHECK_M = None
    THREE_DATA_LEN = "length of m_list_for_azimuthal_check is wrong"


# define the input parameter values
class InputValues(EVs):
    """Defines the input parameter values for our tests."""
    
    TWO_DATA = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 3)],
         'conj': None, 'm_list_for_azimuthal_check': None,
         'div_by_sin': False, 'mul_with_mu': False,
          'nr_omp_threads': 1,
         'use_parallel': False}
    )
    TWO_DATA_DIV_SIN = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 3)],
         'conj': None, 'm_list_for_azimuthal_check': None,
         'div_by_sin': True, 'mul_with_mu': False,
          'nr_omp_threads': 1,
         'use_parallel': False}
    )
    TWO_DATA_MU = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 3)],
         'conj': None, 'm_list_for_azimuthal_check': None,
         'div_by_sin': False, 'mul_with_mu': True,
          'nr_omp_threads': 1,
         'use_parallel': False}
    )
    TWO_DATA_PARALLEL = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 3)],
         'conj': None, 'm_list_for_azimuthal_check': None,
         'div_by_sin': False, 'mul_with_mu': False,
          'nr_omp_threads': 4,
         'use_parallel': True}
    )
    TWO_DATA_CONJ = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 3)],
         'conj': [True, False], 'm_list_for_azimuthal_check': None,
         'div_by_sin': False, 'mul_with_mu': False,
          'nr_omp_threads': 1,
         'use_parallel': False}
    )
    TWO_DATA_CHECK_M = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 3)],
         'conj': None,
         'div_by_sin': False, 'mul_with_mu': False,
         'm_list_for_azimuthal_check': [2,-1,-1], 'nr_omp_threads': 1,
         'use_parallel': False}
    )
    TWO_DATA_LEN = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 3)],
         'conj': None,
         'div_by_sin': False, 'mul_with_mu': False,
         'm_list_for_azimuthal_check': [1,1,1,1,1], 'nr_omp_threads': 1,
         'use_parallel': False}
    )
    THREE_DATA = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 4)],
         'conj': None, 'm_list_for_azimuthal_check': None,
         'div_by_sin': False, 'mul_with_mu': False,
          'nr_omp_threads': 1,
         'use_parallel': False}
    )
    THREE_DATA_DIV_SIN = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 4)],
         'conj': None, 'm_list_for_azimuthal_check': None,
         'div_by_sin': True, 'mul_with_mu': False,
          'nr_omp_threads': 1,
         'use_parallel': False}
    )
    THREE_DATA_MU = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 4)],
         'conj': None, 'm_list_for_azimuthal_check': None,
         'div_by_sin': False, 'mul_with_mu': True,
          'nr_omp_threads': 1,
         'use_parallel': False}
    )
    THREE_DATA_PARALLEL = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 4)],
         'conj': None, 'm_list_for_azimuthal_check': None,
         'div_by_sin': False, 'mul_with_mu': False,
          'nr_omp_threads': 4,
         'use_parallel': True}
    )
    THREE_DATA_CONJ = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 4)],
         'conj': [True, False, False], 'm_list_for_azimuthal_check': None,
         'div_by_sin': False, 'mul_with_mu': False,
          'nr_omp_threads': 1,
         'use_parallel': False}
    )
    THREE_DATA_CHECK_M = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 4)],
         'conj': None,
         'div_by_sin': False, 'mul_with_mu': False,
         'm_list_for_azimuthal_check': [-2,1,1], 'nr_omp_threads': 1,
         'use_parallel': False}
    )
    THREE_DATA_LEN = json.dumps(
        {'descrips': [f"my_data_arr{_i}" for _i in range(1, 4)],
         'conj': None,
         'div_by_sin': False, 'mul_with_mu': False,
         'm_list_for_azimuthal_check': [1,1,1,1,1,1,1], 'nr_omp_threads': 1,
         'use_parallel': False}
    )
