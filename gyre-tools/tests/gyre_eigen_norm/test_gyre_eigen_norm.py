"""Python pytest module for the GYREEigenFunctionNormalizer class.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import pytest
import numpy as np

# import the class that needs to be tested
from gyre_eigen_norm import GYREEigenFunctionNormalizer

# load the mock-up objects
# - extra enumeration object necessary for module
from enumeration_files.coupler_enum_mock import MockEnumCoupler as MeC

# - data object
from data_object_mock import MockDataObj as MDo

# - input data
from enumeration_files.mock_input_enum import (
    InputDataEnums as IDe,
)  # input data type
from enumeration_files.mock_input_enum import (
    InputDataOnes as IDo,
)  # input data 1

# - output data
from enumeration_files.mock_output_enum import OutputData as Od  # output data


# generate the fixture that will be used for the class setup
@pytest.fixture(scope='class', params=[(IDo, 'lee2012'), (IDo, 'schenk2002')])
def setup_test_class(request):
    """Generates a fixture for the testing class so that information is available for testing."""
    # set the data class variable
    request.cls.my_data_class = MDo(request.param[0])
    # set the input class
    request.cls.input_class = request.param[0]
    # set the enumeration class variable
    request.cls.my_enum_class = MeC
    # set the expected output class
    request.cls.expected_output = Od
    # set the normalization convention
    request.cls.norm_convention = request.param[1]


# generate the test class
@pytest.mark.usefixtures('setup_test_class')
class TestGYREEigenFunctionNormalizer:
    """Python test class for the GYRE eigenfunction normalizer."""

    # attribute type declarations
    my_data_class: MDo
    input_class: IDe
    my_enum_class: MeC
    expected_output: Od
    norm_convention: str
    param: tuple

    def test_eigen_function_normalization(self):
        """Test for correct eigenfunction normalization."""
        # initialize the normalizer object
        _norm_object = GYREEigenFunctionNormalizer(
            my_obj=self.my_data_class,
            coupler_enumeration=self.my_enum_class,
            nr_modes=self.my_data_class._nr_modes,
            norm_convention=self.norm_convention,
        )
        # compute the normalization factors
        _norm_object()
        # get computed values
        _comp = _norm_object.normalizing_factors
        # get the expected output values
        _out = self.expected_output.get_output(
            self.input_class, self.norm_convention
        )
        # TODO: compute on paper! --> I COMPUTED WITH MATHEMATICA...
        # assert the output is as required
        assert np.allclose(_comp, _out)
