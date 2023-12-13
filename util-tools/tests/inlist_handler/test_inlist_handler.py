"""Pytest module for the 'inlist_handler' module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import pytest

# import the class to be tested
from inlist_handler import InlistHandler


# session fixture - create temporary inlist file + defaults
@pytest.fixture(scope='session')
def inlist_file(tmp_path_factory):
    # generate a path to a temporary inlist file
    my_path = tmp_path_factory.mktemp('my_data') / 'test.in'
    my_path_default = tmp_path_factory.getbasetemp() / 'defaults/'
    my_path_default.mkdir(parents=True, exist_ok=True)
    my_path_default = my_path_default / 'test.defaults'
    # my_path_default = tmp_path_factory.mktemp("defaults") / "test.defaults"
    # construct the text for the inlist and defaults file
    my_text = """
        % Test inlist
        % Author: John Doe
        
        %%%%%%%%%%%%%%%%%%%%%%%%%
        % set of test arguments %
        %%%%%%%%%%%%%%%%%%%%%%%%%
        % bool load checks
        my_bool = True
        my_capital_bool = FALSE
        % int load check
        my_int = 8
        % list load check
        my_list = [2,1,1,1,2]
        my_list_second = [8.0,7.0,6.0]
        % string load check
        my_string = 'this is a string'
        my_string_second = '/path/to/somewhere'
        % float load check
        my_float = 8.0
        % generic comment load check
        my_comment_load = 8%7%6
        """
    # write text to the temporary inlist file
    my_path.write_text(my_text)
    my_path_default.write_text(my_text)
    # return the path (only need the path to the inlist,
    # path to defaults file is automatically generated)
    return my_path


# fixture
@pytest.fixture(scope='class')
def setup_test_class(request):
    # store the expected output
    request.cls.expected_output = {
        'my_bool': True,
        'my_capital_bool': False,
        'my_int': 8,
        'my_list': [2, 1, 1, 1, 2],
        'my_list_second': [8.0, 7.0, 6.0],
        'my_string': 'this is a string',
        'my_string_second': '/path/to/somewhere',
        'my_float': 8.0,
        'my_comment_load': 8,
    }


# pytest class
@pytest.mark.usefixtures('setup_test_class')
class TestTomlInlistHandler:
    """Tests the TomlInlistHandler class"""

    # attribute type declarations
    # my_file_path: pathlib.Path
    expected_output: dict

    def test_toml_read(self, inlist_file):
        """Test whether the toml input info can be read."""
        # read Toml input from the temporary test file
        my_output = InlistHandler.get_inlist_values(
            inlist_path=str(inlist_file.resolve())
        )
        # assert the output is ok
        assert my_output == self.expected_output
