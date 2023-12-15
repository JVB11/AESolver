'''Pytest module used to test the reading of MESA profile files.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
# import support modules
from aesolver.model_handling.enumeration_files.enumeration_models \
    import MESAProfileFiles
from aesolver.model_handling.subselection_classes import \
    MESAProfileSubSelector
from compare_dicts import compare_mesa_profile
# import mocking functions
from mock_mesa_profile import create_header_mesa, create_body_mesa
from mock_mesa_profile import ExpectedMESAProfileData as EMPd
# import the classes to be tested
from aesolver.model_handling.read_models import ModelReader


# session fixture - create temporary MESA profile file
@pytest.fixture(scope='session')
def mesa_profile_file(tmp_path_factory) -> str:
    # generate a path to a temporary MESA profile file
    my_path = tmp_path_factory.mktemp("my_MESA_profile")
    my_path_to_file = my_path / "my_substring_another_M6_00.dat"
    # get the information for the temporary MESA profile file
    my_header, nr_zones = create_header_mesa()
    my_array_data = create_body_mesa(nz=nr_zones)
    # create the temporary MESA profile file
    with open(str(my_path_to_file), 'w') as my_f:
        my_f.write(my_header)
        my_f.write('\n')
        my_f.write(my_array_data)
    # return the path to the file
    return my_path


# fixture 
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    # initialize the ModelReader object
    request.cls.reader: ModelReader = \
        ModelReader(MESAProfileFiles)
    # store the additional substrings used to load the file
    request.cls.substrings: tuple[str] | str \
        = ('my_substring', 'another', 'M6')
    # store the substring selecting method object
    request.cls.select_obj: MESAProfileSubSelector = \
        MESAProfileSubSelector(
            mesa_file_suffix='dat'
            )
    
    
# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestReadMESAProfile:
    """Tests the model reader for MESA profile files."""
    
    # attribute type declaration
    reader: ModelReader
    substrings: str
    select_obj: MESAProfileSubSelector    
    
    # use the ModelReader class to read the data file
    def test_dat_read(self, mesa_profile_file: str) -> None:
        """Use the ModelReader class to read the MESA .dat file.
        """
        # read the file
        my_data_list = self.reader.data_file_reader(
            my_dir=mesa_profile_file,
            additional_substring=self.substrings,
            subselection_method=self.select_obj.subselect,
            mode_number=1, structure_file=True, g_mode=None
            )
        # perform a comparison - need to convert types because
        # they are transformed to strings.. TODO: stop this conversion
        # at this stage?
        comparison_bool = compare_mesa_profile(
            expected_mesa=EMPd.get_dict(),
            vals_mesa=my_data_list[0]
            )
        # assert the comparison worked
        assert comparison_bool
