"""Pytest module for the 'toml_inlist_handler' module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import pytest

# import the class to be tested
from inlist_handler import TomlInlistHandler


# session fixture - create temporary toml file
@pytest.fixture(scope='session')
def toml_file(tmp_path_factory):
    # generate a path to a temporary toml file
    my_path = tmp_path_factory.mktemp('my_data') / 'test.toml'
    # write text to the temporary toml file
    my_path.write_text(
        """
        [my-table]
        my-version = "3.11"
        my-implementation = "TEST"
        my-none = {}
        """
    )
    # return the path
    return my_path


# fixture
@pytest.fixture(scope='class')
def setup_test_class(request):
    # store the expected output
    request.cls.expected_output = {
        'my-table': {
            'my-version': '3.11',
            'my-implementation': 'TEST',
            'my-none': None,
        }
    }


# pytest class
@pytest.mark.usefixtures('setup_test_class')
class TestTomlInlistHandler:
    """Tests the TomlInlistHandler class"""

    # attribute type declarations
    expected_output: dict

    def test_toml_read(self, toml_file):
        """Test whether the toml input info can be read."""
        # read Toml input from the temporary test file
        my_output = TomlInlistHandler.get_inlist_values(inlist_path=toml_file)
        # assert the output is ok
        assert my_output == self.expected_output
