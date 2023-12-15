'''Pytest module used to test the functionalities of the ModelLoader class.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
# import mocking functions
from mock_gyre_detail import \
    create_mock_gyre_hdf5_mode_file
from mock_mesa_profile import \
    create_header_mesa, create_body_mesa
from mock_polytrope_modes import \
    create_mock_gyre_hdf5_polytrope_mode_file
from mock_polytrope_structure import \
    create_mock_gyre_hdf5_polytrope_structure_file
# import support modules for testing
from mock_mesa_profile import \
    ExpectedMESAProfileData as EMPd
from compare_dicts import \
    compare_dicts, compare_mesa_profile
from mock_gyre_detail import \
    GYREDetailExpectedValuesAfterRead as GDEVaR
from mock_polytrope_structure import \
    GYREPolytropeStructureExpectedValuesAfterLoad \
        as GPSeVaL
from mock_polytrope_modes import \
    GYREPolytropeModeExpectedValuesAfterRead \
        as GPMeVaR
# import the classes to be tested
from aesolver.model_handling.load_models \
    import ModelLoader


# session fixture - create temporary 
# GYRE details and MESA profile files
@pytest.fixture(scope='session')
def my_mode_files(tmp_path_factory) -> str:
    # create the path to the base directory
    base_path = tmp_path_factory.mktemp("my_base_dir")
    # create the temporary GYRE directory and files
    gyre_dir = base_path / "GYRE/"
    gyre_dir.mkdir(parents=True, exist_ok=True)
    my_path_to_gyre_file1 = \
        gyre_dir / "my_substring_g_mode_l2m+2n+23.h5"
    my_path_to_gyre_file2 = \
        gyre_dir / "my_substring_g_mode_l2m+2n+24.h5"
    my_path_to_gyre_file3 = \
        gyre_dir / "my_substring_g_mode_l2m+2n+25.h5"
    create_mock_gyre_hdf5_mode_file(
        my_path_to_gyre_file1, mode_n=23
        )
    create_mock_gyre_hdf5_mode_file(
        my_path_to_gyre_file2, mode_n=24
        )
    create_mock_gyre_hdf5_mode_file(
        my_path_to_gyre_file3, mode_n=25
        )
    # create the temporary MESA directory and profile file
    mesa_dir = base_path / "MESA/"
    mesa_dir.mkdir(parents=True, exist_ok=True)
    my_mesa_path = \
        mesa_dir / "my_substring_another_M6_00.dat"
    my_header, nz = create_header_mesa()
    my_body = create_body_mesa(nz=nz)
    with open(str(my_mesa_path), 'w') as my_mesa_file:
        my_mesa_file.write(my_header)
        my_mesa_file.write('\n')
        my_mesa_file.write(my_body)
    # create the temporary polytrope structure file
    poly_struct_dir = base_path / "POLYTROPE_MODEL/"
    poly_struct_dir.mkdir(parents=True, exist_ok=True)
    my_poly_struct_path = \
        poly_struct_dir / "my_substring_another_M6_00.h5"
    create_mock_gyre_hdf5_polytrope_structure_file(
        my_poly_struct_path
        )
    # create temporary polytrope structure file
    # - alternative in different directory
    poly_struct_dir2 = base_path / "SOME/OTHER/PATH/"
    poly_struct_dir2.mkdir(parents=True, exist_ok=True)
    my_poly_struct_path2 = \
        poly_struct_dir2 / "my_substring_another_M6_00.h5"
    create_mock_gyre_hdf5_polytrope_structure_file(
        my_poly_struct_path2
        )    
    # create the temporary polytrope mode files
    poly_mode_dir = base_path / "POLYTROPE_OSCILLATION/"
    poly_mode_dir.mkdir(parents=True, exist_ok=True)
    my_path_to_poly_mode_file1 = \
        poly_mode_dir / "my_substring_g_mode_l2m+2n+23.h5"
    my_path_to_poly_mode_file2 = \
        poly_mode_dir / "my_substring_g_mode_l2m+2n+24.h5"
    my_path_to_poly_mode_file3 = \
        poly_mode_dir / "my_substring_g_mode_l2m+2n+25.h5"
    create_mock_gyre_hdf5_polytrope_mode_file(
        my_path_to_poly_mode_file1, n_g=23
        )
    create_mock_gyre_hdf5_polytrope_mode_file(
        my_path_to_poly_mode_file2, n_g=24
        )
    create_mock_gyre_hdf5_polytrope_mode_file(
        my_path_to_poly_mode_file3, n_g=25
        ) 
    # create the temporary polytrope mode files
    # - alternatives in different directories
    poly_mode_dir2 = base_path / "YET/ANOTHER/PATH/"
    poly_mode_dir2.mkdir(parents=True, exist_ok=True)
    my_path_to_poly2_mode_file1 = \
        poly_mode_dir2 / "my_substring_g_mode_l2m+2n+23.h5"
    my_path_to_poly2_mode_file2 = \
        poly_mode_dir2 / "my_substring_g_mode_l2m+2n+24.h5"
    my_path_to_poly2_mode_file3 = \
        poly_mode_dir2 / "my_substring_g_mode_l2m+2n+25.h5"
    create_mock_gyre_hdf5_polytrope_mode_file(
        my_path_to_poly2_mode_file1, n_g=23
        )
    create_mock_gyre_hdf5_polytrope_mode_file(
        my_path_to_poly2_mode_file2, n_g=24
        )
    create_mock_gyre_hdf5_polytrope_mode_file(
        my_path_to_poly2_mode_file3, n_g=25
        )
    # return the base directory path
    return base_path, poly_struct_dir2, poly_mode_dir2


# fixture for the test class
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    # store the output (sub-)directories
    request.cls.mesa_output_dir: str | None = None
    request.cls.gyre_output_dir: str | None = None
    # store the additional search substrings used to
    # load the MESA and GYRE files
    request.cls.gyre_substring: tuple[str] | str = \
        "my_substring"
    request.cls.mesa_substrings: tuple[str] | str = \
        ('my_substring', 'another', 'M6')
    # store the additional search substrings used to
    # load the polytrope structure and mode files
    request.cls.poly_mode_substring: tuple[str] | str = \
        "my_substring"
    request.cls.poly_struct_substring: tuple[str] | str = \
        ("my_substring", "another", "M6")
    # store the mesa file suffix
    request.cls.mesa_suffix: str = 'dat'  
    # store the polytrope structure and model suffixes
    request.cls.poly_struct_suffix: str = 'h5'
    # store the search quantum number for
    # the GYRE detail files
    request.cls.search_l: list[int] = [2, 2, 2]
    request.cls.search_m: list[int] = [2, 2, 2]
    request.cls.search_n: list[int] = [23, 24, 25]
    # additional input options
    request.cls.infer_dir: bool = True
    request.cls.unit_system: str = 'CGS'


# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestModelLoader:
    """Tests the 'ModelLoader' class functionalities."""
    
    # attribute type declaration
    mesa_output_dir: str | None
    gyre_output_dir: str | None
    gyre_substring: tuple[str] | str
    mesa_substrings: tuple[str] | str
    poly_mode_substring: tuple[str] | str
    poly_struct_substring: tuple[str] | str
    mesa_suffix: str
    poly_struct_suffix: str
    search_l: list[int]
    search_m: list[int]
    search_n: list[int]
    infer_dir: bool
    unit_system: str
    
    # test the initialization
    def test_initialization(self, my_mode_files: list[str]) -> None:
        """Tests the initialization of the ModelLoader module.
        
        Parameters
        ----------
        my_mode_files : list[str]
            Path to the base directory of the mode files, as well as the alternative directory paths.
        """
        # create a ModelLoader object
        my_obj = ModelLoader(
            base_dir=my_mode_files[0],
            mesa_output_dir=self.mesa_output_dir,
            gyre_output_dir=self.gyre_output_dir,
            infer_dir=self.infer_dir,
            search_l=self.search_l,
            search_m=self.search_m,
            search_n=self.search_n,
            unit_system=self.unit_system,
            gyre_detail_substring=self.gyre_substring,
            mesa_profile_substrings=self.mesa_substrings,
            mesa_suffix=self.mesa_suffix,
            polytrope_model_suffix=self.poly_struct_suffix,
            polytrope_model_substrings=self.poly_struct_substring,
            polytrope_oscillation_substrings=self.poly_mode_substring
            )
        # check if it was loaded
        assert my_obj
    
    # test the loading method for the MESA profile
    def test_loading_mesa(self, my_mode_files: list[str]) -> None:
        """Tests the MESA profile loading method of ModelLoader.

        Parameters
        ----------
        my_mode_files : list[str]
            Path to the base directory of the mode files, as well as the alternative directory paths.
        """
        # create a ModelLoader object
        my_obj = ModelLoader(
            base_dir=my_mode_files[0],
            mesa_output_dir=self.mesa_output_dir,
            gyre_output_dir=self.gyre_output_dir,
            infer_dir=self.infer_dir,
            search_l=self.search_l,
            search_m=self.search_m,
            search_n=self.search_n,
            unit_system=self.unit_system,
            gyre_detail_substring=self.gyre_substring,
            mesa_profile_substrings=self.mesa_substrings,
            mesa_suffix=self.mesa_suffix,
            polytrope_model_suffix=self.poly_struct_suffix,
            polytrope_model_substrings=self.poly_struct_substring,
            polytrope_oscillation_substrings=self.poly_mode_substring
            )
        # load the MESA profile file
        my_obj.read_data_files(mesa_profile=True)
        # assert if the correct data is loaded
        assert compare_mesa_profile(
            expected_mesa=EMPd.get_dict(),
            vals_mesa=my_obj.mesa_profile[0]
            )

    # test the loading method for the GYRE detail files
    def test_loading_gyre(self, my_mode_files: list[str]) -> None:
        """Tests the MESA profile loading method of ModelLoader.

        Parameters
        ----------
        my_mode_files : list[str]
            Path to the base directory of the mode files, as well as the alternative directory paths.
        """
        # create a ModelLoader object
        my_obj = ModelLoader(
            base_dir=my_mode_files[0],
            mesa_output_dir=self.mesa_output_dir,
            gyre_output_dir=self.gyre_output_dir,
            infer_dir=self.infer_dir,
            search_l=self.search_l,
            search_m=self.search_m,
            search_n=self.search_n,
            unit_system=self.unit_system,
            gyre_detail_substring=self.gyre_substring,
            mesa_profile_substrings=self.mesa_substrings,
            mesa_suffix=self.mesa_suffix,
            polytrope_model_suffix=self.poly_struct_suffix,
            polytrope_model_substrings=self.poly_struct_substring,
            polytrope_oscillation_substrings=self.poly_mode_substring
            )
        # load the GYRE detail files
        my_obj.read_data_files(gyre_detail=True)
        # check if all GYRE files are loaded correctly
        for _i, _d in enumerate(my_obj.gyre_details):
            assert compare_dicts(
                _d, GDEVaR.get_dict(mode_nr=_i + 1)
                )

    # test the loading method for the
    # GYRE polytrope structure files
    def test_loading_poly_struct(self, my_mode_files: list[str]) -> None:
        """Tests the GYRE polytrope structure model loading method of ModelLoader.

        Parameters
        ----------
        my_mode_files : list[str]
            Path to the base directory of the mode files, as well as the alternative directory paths.
        """
        # create a ModelLoader object
        my_obj = ModelLoader(
            base_dir=my_mode_files[0],
            mesa_output_dir=self.mesa_output_dir,
            gyre_output_dir=self.gyre_output_dir,
            infer_dir=self.infer_dir,
            search_l=self.search_l,
            search_m=self.search_m,
            search_n=self.search_n,
            unit_system=self.unit_system,
            gyre_detail_substring=self.gyre_substring,
            mesa_profile_substrings=self.mesa_substrings,
            mesa_suffix=self.mesa_suffix,
            polytrope_model_suffix=self.poly_struct_suffix,
            polytrope_model_substrings=self.poly_struct_substring,
            polytrope_oscillation_substrings=self.poly_mode_substring
            )
        # load the polytrope structure file
        my_obj.read_data_files(polytrope_model=True,
                               polytrope_mass=3.0,
                               polytrope_radius=5.0)  # TEST MASS/RADIUS
        # check if the polytrope structure file is loaded correctly
        assert compare_dicts(
            my_obj.polytrope_model[0],
            GPSeVaL.get_dict()
            )       

    # test the loading method for the
    # GYRE polytrope mode files
    def test_loading_poly_modes(self, my_mode_files: list[str]) -> None:
        """Tests the GYRE polytrope modes loading method of ModelLoader.

        Parameters
        ----------
        my_mode_files : list[str]
            Path to the base directory of the mode files, as well as the alternative directory paths.
        """
        # create a ModelLoader object
        my_obj = ModelLoader(
            base_dir=my_mode_files[0],
            mesa_output_dir=self.mesa_output_dir,
            gyre_output_dir=self.gyre_output_dir,
            infer_dir=self.infer_dir,
            search_l=self.search_l,
            search_m=self.search_m,
            search_n=self.search_n,
            unit_system=self.unit_system,
            gyre_detail_substring=self.gyre_substring,
            mesa_profile_substrings=self.mesa_substrings,
            mesa_suffix=self.mesa_suffix,
            polytrope_model_suffix=self.poly_struct_suffix,
            polytrope_model_substrings=self.poly_struct_substring,
            polytrope_oscillation_substrings=self.poly_mode_substring
            )
        # load the polytrope model files
        my_obj.read_data_files(polytrope_oscillation=True)
        # check if all polytrope model files are loaded correctly
        for _i, _d in enumerate(my_obj.polytrope_oscillation_models):
            assert compare_dicts(
                _d, GPMeVaR.get_dict(mode_nr=_i + 1)
                )       

    # test the loading method for the alternative
    # GYRE polytrope structure and mode files
    def test_alternative_paths_poly(self, my_mode_files: list[str]) -> None:
        """Tests the GYRE polytrope modes and polytrope structure file loading methods of ModelLoader using alternative directories.

        Parameters
        ----------
        my_mode_files : list[str]
            Path to the base directory of the mode files, as well as the alternative directory paths.
        """
        # create a ModelLoader object
        my_obj = ModelLoader(
            base_dir=my_mode_files[0],
            mesa_output_dir=self.mesa_output_dir,
            gyre_output_dir=self.gyre_output_dir,
            infer_dir=False,
            search_l=self.search_l,
            search_m=self.search_m,
            search_n=self.search_n,
            unit_system=self.unit_system,
            gyre_detail_substring=self.gyre_substring,
            mesa_profile_substrings=self.mesa_substrings,
            mesa_suffix=self.mesa_suffix,
            polytrope_model_suffix=self.poly_struct_suffix,
            polytrope_model_substrings=self.poly_struct_substring,
            polytrope_oscillation_substrings=self.poly_mode_substring,
            polytrope_model_output_dir=my_mode_files[1],
            polytrope_oscillation_output_dir=my_mode_files[2]
            )
        # load the polytrope model files
        my_obj.read_data_files(polytrope_model=True,
                               polytrope_oscillation=True,
                               polytrope_mass=3.0,
                               polytrope_radius=5.0)  # TEST MASS/RADIUS)
        # check if all polytrope model files are loaded correctly
        for _i, _d in enumerate(my_obj.polytrope_oscillation_models):
            assert compare_dicts(
                _d, GPMeVaR.get_dict(mode_nr=_i + 1)
                )
        # check if the polytrope structure file is loaded correctly
        assert compare_dicts(
            my_obj.polytrope_model[0],
            GPSeVaL.get_dict()
            ) 
        
    # test the loading method for the alternative
    # GYRE polytrope structure and mode files
    @pytest.mark.xfail(reason='Polytrope directories not passed and not inferred.')
    def test_alternative_paths_poly(self, my_mode_files: list[str]) -> None:
        """Tests the GYRE polytrope modes and polytrope structure file loading methods of ModelLoader using alternative directories.

        Parameters
        ----------
        my_mode_files : list[str]
            Path to the base directory of the mode files, as well as the alternative directory paths.
        """
        # create a ModelLoader object
        my_obj = ModelLoader(
            base_dir=my_mode_files[0],
            mesa_output_dir=self.mesa_output_dir,
            gyre_output_dir=self.gyre_output_dir,
            infer_dir=False,
            search_l=self.search_l,
            search_m=self.search_m,
            search_n=self.search_n,
            unit_system=self.unit_system,
            gyre_detail_substring=self.gyre_substring,
            mesa_profile_substrings=self.mesa_substrings,
            mesa_suffix=self.mesa_suffix,
            polytrope_model_suffix=self.poly_struct_suffix,
            polytrope_model_substrings=self.poly_struct_substring,
            polytrope_oscillation_substrings=self.poly_mode_substring
            )
        # load the polytrope model files
        my_obj.read_data_files(polytrope_model=True,
                               polytrope_oscillation=True,
                               polytrope_mass=3.0,
                               polytrope_radius=5.0)  # TEST MASS/RADIUS)
        # check if all polytrope model files are loaded correctly
        for _i, _d in enumerate(my_obj.polytrope_oscillation_models):
            assert compare_dicts(
                _d, GPMeVaR.get_dict(mode_nr=_i + 1)
                )
        # check if the polytrope structure file is loaded correctly
        assert compare_dicts(
            my_obj.polytrope_model[0],
            GPSeVaL.get_dict()
            )        
