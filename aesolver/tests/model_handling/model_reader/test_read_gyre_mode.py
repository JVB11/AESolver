'''Pytest module used to test the reading of GYRE summary files.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
# import support modules
from aesolver.model_handling.enumeration_files.enumeration_models \
    import GYREDetailFiles
from aesolver.model_handling.subselection_classes import \
    GYREDetailSubSelector
from compare_dicts import compare_dicts
# import mocking functions
from mock_gyre_detail import \
    create_mock_gyre_hdf5_mode_file
from mock_gyre_detail import \
    GYREDetailExpectedValuesAfterRead as GDeVaR
# import the classes to be tested
from aesolver.model_handling.read_models import ModelReader


# session fixture - create temporary GYRE details files
@pytest.fixture(scope='session')
def gyre_mode_file(tmp_path_factory) -> str:
    # TODO: at this moment, it needs all THREE FILES TO BE PRESENT
    # IN ORDER TO LOAD --> remove this dependency!
    # generate a path to a temporary GYRE mode file
    my_path = tmp_path_factory.mktemp("my_GYRE_mode")
    my_path_to_file = my_path / "my_substring_g_mode_l2m+2n+25.h5"
    my_path_to_file2 = my_path / "my_substring_g_mode_l2m+2n+24.h5"
    my_path_to_file3 = my_path / "my_substring_g_mode_l2m+2n+23.h5"
    # create the temporary HDF5 file
    create_mock_gyre_hdf5_mode_file(
        my_path_to_file, mode_nr=1
        )
    create_mock_gyre_hdf5_mode_file(
        my_path_to_file2, mode_nr=2
        )
    create_mock_gyre_hdf5_mode_file(
        my_path_to_file3, mode_nr=3
        )
    # return the paths to the file
    return my_path


# fixture
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    # initialize the ModelReader object
    request.cls.reader: ModelReader = \
        ModelReader(GYREDetailFiles)
    # store the additional substring used to load the file
    request.cls.substring: str = "my_substring"
    # store the substring selecting method object
    request.cls.select_obj: GYREDetailSubSelector = \
        GYREDetailSubSelector(
            search_l=[2, 2, 2],
            search_m=[2, 2, 2],
            search_n=[25, 24, 23]
            )


# define the test class
# TODO: add additional tests for the keywords!
@pytest.mark.usefixtures('setup_test_class')
class TestReadGYREDetail:
    """Tests the model reader for GYRE detail files."""
    
    # attribute type declaration
    reader: ModelReader
    substring: str
    select_obj: GYREDetailSubSelector
    
    # use the ModelReader class to read the data file
    def test_hdf5_read(self, gyre_mode_file: str) -> None:
        """Use the ModelReader class to read the HDF5 GYRE Detail file."""
        # read the file -- TODO: don't know why I am returning list
        # as this only reads the data from a single object? --> FIX
        my_data_list = self.reader.data_file_reader(
            my_dir=gyre_mode_file,
            additional_substring=self.substring,
            subselection_method=self.select_obj.subselect,
            mode_number=1, structure_file=False, g_mode=None
            )
        # check if the files are read by comparing with
        # the expected values
        assert compare_dicts(
            my_data_list[0], GDeVaR.get_dict(mode_nr=1)
            )
        # read the file -- TODO: don't know why I am returning list
        # as this only reads the data from a single object? --> FIX
        my_data_list2 = self.reader.data_file_reader(
            my_dir=gyre_mode_file,
            additional_substring=self.substring,
            subselection_method=self.select_obj.subselect,
            mode_number=2, structure_file=False, g_mode=None
            )
        # check if the files are read by comparing with
        # the expected values
        assert compare_dicts(
            my_data_list2[0], GDeVaR.get_dict(mode_nr=2)
            )
        # read the file -- TODO: don't know why I am returning list
        # as this only reads the data from a single object? --> FIX
        my_data_list3 = self.reader.data_file_reader(
            my_dir=gyre_mode_file,
            additional_substring=self.substring,
            subselection_method=self.select_obj.subselect,
            mode_number=3, structure_file=False, g_mode=None
            )
        # check if the files are read by comparing with
        # the expected values
        assert compare_dicts(
            my_data_list3[0], GDeVaR.get_dict(mode_nr=3)
            )
        