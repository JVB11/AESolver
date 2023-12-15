'''Pytest module used to test the reading of GYRE polytrope mode files.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
# import support modules
from aesolver.model_handling.enumeration_files.enumeration_models \
    import PolytropeOscillationModelFiles
from aesolver.model_handling.subselection_classes import \
    PolytropeOscillationModelSubSelector
from compare_dicts import compare_dicts
# import mocking function/class
from mock_polytrope_mode_file import \
    create_mock_gyre_hdf5_polytrope_mode_file
from mock_polytrope_mode_file import \
    GYREPolytropeModeExpectedValuesAfterRead as GPMeVaR
# import the class to be tested
from aesolver.model_handling.read_models import ModelReader


# session fixture - create temporary GYRE polytrope mode files
@pytest.fixture(scope='session')
def polytrope_mode_files(tmp_path_factory) -> str:
    # generate paths to temporary GYRE polytrope mode files
    my_path = tmp_path_factory.mktemp("my_polytrope_modes")
    my_path_to_file = my_path / "my_substring_g_mode_l2m+2n+23.h5"
    my_path_to_file2 = my_path / "my_substring_g_mode_l2m+1n+24.h5"
    my_path_to_file3 = my_path / "my_substring_g_mode_l2m+1n+25.h5"
    # create input dictionaries
    my_i_dict = {'l': 2, 'm': 2, 'n_g': 23, 'omega': (0.75, 0.0)}
    my_i_dict2 = {'l': 2, 'm': 1, 'n_g': 24, 'omega': (0.65, 0.0)}
    my_i_dict3 = {'l': 2, 'm': 1, 'n_g': 25, 'omega': (0.15, 0.0)}
    # create the temporary files
    create_mock_gyre_hdf5_polytrope_mode_file(
        my_path_to_file, my_i_dict
        )
    create_mock_gyre_hdf5_polytrope_mode_file(
        my_path_to_file2, my_i_dict2
        )
    create_mock_gyre_hdf5_polytrope_mode_file(
        my_path_to_file3, my_i_dict3
        )
    # create extra nuisance files that need not be loaded
    my_path_to_nuisance_file = \
        my_path / "my_substring_g_mode_l3m+2n+23.h5"
    my_path_to_nuisance_file2 = \
        my_path / "my_substring_g_mode_l2m+1n+37.h5"
    my_path_to_nuisance_file3 = \
        my_path / "my_substring_g_mode_l2m+2n+37.h5"
    my_path_to_nuisance_file4 = \
        my_path / "my_substring_g_mode_l2m+0n+25.h5"
    my_n_dict = {'l': 3, 'm': 2, 'n_g': 23, 'omega': (0.75, 0.0)}
    my_n_dict2 = {'l': 2, 'm': 1, 'n_g': 37, 'omega': (0.75, 0.0)}
    my_n_dict3 = {'l': 2, 'm': 2, 'n_g': 37, 'omega': (0.75, 0.0)}
    my_n_dict4 = {'l': 2, 'm': 0, 'n_g': 25, 'omega': (0.75, 0.0)}
    create_mock_gyre_hdf5_polytrope_mode_file(
        my_path_to_nuisance_file, my_n_dict
        )
    create_mock_gyre_hdf5_polytrope_mode_file(
        my_path_to_nuisance_file2, my_n_dict2
        )
    create_mock_gyre_hdf5_polytrope_mode_file(
        my_path_to_nuisance_file3, my_n_dict3
        )
    create_mock_gyre_hdf5_polytrope_mode_file(
        my_path_to_nuisance_file4, my_n_dict4
        )
    # return the path to the directory containing the files
    return my_path


# fixture used to set up test class
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    """Setup fixture for the test class."""
    # initialize the ModelReader object
    request.cls.reader: ModelReader = \
        ModelReader(PolytropeOscillationModelFiles)
    # store the additional substring used to load the file
    request.cls.substring: str = 'my_substring'
    # store the substring selecting method object
    request.cls.select_obj: \
        PolytropeOscillationModelSubSelector = \
            PolytropeOscillationModelSubSelector(
            search_l=[2, 2, 2],
            search_m=[2, 1, 1],
            search_n=[23, 24, 25]                
            )


# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestReadPolytropeStructureFile:
    """Tests the model reader for GYRE polytrope structure files."""
    
    # attribute type declaration
    reader: ModelReader
    substring: str
    select_obj: PolytropeOscillationModelSubSelector
    
    # use the ModelReader class to read the data file
    def test_hdf5_read(
        self, polytrope_mode_files: str
        ) -> None:
        """Use the ModelReader class to read the HDF5 GYRE polytrope structure file.

        Parameters
        ----------
        polytrope_mode_files : str
            Path to the directory containing the  GYRE polytrope mode files.
        """        
        # read the file
        my_data_list = self.reader.data_file_reader(
            my_dir=polytrope_mode_files,
            additional_substring=self.substring,
            subselection_method=self.select_obj.subselect,
            mode_number=1, structure_file=False, g_mode=None
            )
        # perform comparison
        assert compare_dicts(
            first=my_data_list[0],
            second=GPMeVaR.get_dict(mode_nr=1)
            )
        # read the file
        my_data_list2 = self.reader.data_file_reader(
            my_dir=polytrope_mode_files,
            additional_substring=self.substring,
            subselection_method=self.select_obj.subselect,
            mode_number=2, structure_file=False, g_mode=None
            )
        # perform comparison
        assert compare_dicts(
            first=my_data_list2[0],
            second=GPMeVaR.get_dict(mode_nr=2)
            )
        # read the file
        my_data_list3 = self.reader.data_file_reader(
            my_dir=polytrope_mode_files,
            additional_substring=self.substring,
            subselection_method=self.select_obj.subselect,
            mode_number=3, structure_file=False, g_mode=None
            )
        # perform comparison
        assert compare_dicts(
            first=my_data_list3[0],
            second=GPMeVaR.get_dict(mode_nr=3)
            )
