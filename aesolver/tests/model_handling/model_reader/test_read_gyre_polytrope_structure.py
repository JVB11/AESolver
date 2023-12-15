'''Pytest module used to test the reading of GYRE polytrope structure files.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
# import support modules
from aesolver.model_handling.enumeration_files.enumeration_models \
    import PolytropeModelFiles
from aesolver.model_handling.subselection_classes import \
    PolytropeModelSubSelector
from compare_dicts import compare_dicts
# import mocking function/class
from mock_polytrope_structure import \
    create_mock_gyre_hdf5_polytrope_structure_file
from mock_polytrope_structure import \
    GYREPolytropeStructureExpectedValuesAfterRead as \
        GPSeVaR
# import the classes to be tested
from aesolver.model_handling.read_models import ModelReader


# session fixture - create temporary GYRE polytrope structure file
@pytest.fixture(scope='session')
def polytrope_profile_file(tmp_path_factory) -> str:
    # generate a path to a temporary GYRE polytrope structure file
    my_path = tmp_path_factory.mktemp("my_polytrope_structure")
    my_path_to_file = my_path / "my_substring_another_M4_00.h5"    
    # create the temporary file
    create_mock_gyre_hdf5_polytrope_structure_file(my_path_to_file)
    # return the path to the directory containing the file
    return my_path


# fixture used to set up test class
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    """Setup fixture for the test class."""
    # initialize the ModelReader object
    request.cls.reader: ModelReader = \
        ModelReader(PolytropeModelFiles)
    # store the additional substring used to load the file
    request.cls.substring: list[str] = \
        ['my_substring', 'another']
    # store the substring selecting method object
    request.cls.select_obj: PolytropeModelSubSelector = \
        PolytropeModelSubSelector(
            polytrope_model_suffix='h5'
        )
    
    
# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestReadPolytropeStructureFile:
    """Tests the model reader for GYRE polytrope structure files."""
    
    # attribute type declaration
    reader: ModelReader
    substring: list[str]
    select_obj: PolytropeModelSubSelector
    
    # use the ModelReader class to read the data file
    def test_hdf5_read(
        self, polytrope_profile_file: str
        ) -> None:
        """Use the ModelReader class to read the HDF5 GYRE polytrope structure file.

        Parameters
        ----------
        polytrope_profile_file : str
            Path to the GYRE polytrope structure file.
        """
        # read the file
        my_data_list = self.reader.data_file_reader(
            my_dir=polytrope_profile_file,
            additional_substring=self.substring,
            subselection_method=self.select_obj.subselect,
            mode_number=1, structure_file=True, g_mode=None
            )
        # perform comparison
        assert compare_dicts(
            first=my_data_list[0], second=GPSeVaR.get_dict()
            )
        