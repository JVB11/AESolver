'''Python module containing functions that create a mock MESA profile file.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import typing
import numpy as np


# define a base class to generate a MESA formatted float
class MESAFormatNumber:
    def __init__(self, val: float) -> None:
        self.val = val

    def __format__(self, format_spec: str) -> str:
        ss = ('{0:' + format_spec + '}').format(self.val)
        if 'E' in ss:
            mantissa, exp = ss.split('E')
            return mantissa + 'E'+ exp[0] + '0' + exp[1:]
        return ss


# generate a MESA-formatted float
def mesa_float(my_float: float) -> str:
    """Generates a MESA-style formatted float.
    
    Notes
    -----
    No 'first_column' parameter added, since floats do not appear in the first columns of MESA profile files!

    Parameters
    ----------
    my_float : float
        The float that needs to be MESA formatted.
        
    Returns
    -------
    str
        The MESA-formatted float string.
    """
    return '{0:40.16E}'.format(MESAFormatNumber(my_float))
    
    
# generate a MESA-formatted string
def mesa_string(my_string: str, first_row: bool=False) -> str:
    """Generates a MESA-style formatted string.

    Parameters
    ----------
    my_string : str
        The string that needs to be MESA-formatted.
    first_row : bool, optional
        Whether the string appears on the first row of the MESA-file; by default False
    
    Returns
    -------
    str
        The MESA-formatted string.
    """
    if first_row:
        return '{0: >40}'.format(my_string)
    else:
        return '{0: >41}'.format(my_string)


# generate a MESA-formatted int    
def mesa_int(my_int: int, first_row: bool=False) -> str:
    """Generates a MESA-style formatted integer.

    Parameters
    ----------
    my_int : int
        The integer that needs to be MESA-formatted.
    first_row : bool, optional
        Whether the integer appears on the first row of the MESA file; by default False.
        
    Returns
    -------
    str
        The MESA-formatted integer.
    """
    if first_row:
        return '{0:40}'.format(my_int)
    else:
        return '{0:41}'.format(my_int)


def handle_value(my_val: str | int | float, first_row: bool=False) -> str:
    """Handles formatting of a value for the MESA mock
    profile file.

    Parameters
    ----------
    my_val : str or int or float
        Value to be MESA-formatted.
    first_row : bool, optional
        Whether the integer/string appears on the first row of the MESA file; by default False.
        
    Returns
    -------
    str
        The MESA-formatted string of the value.
    """
    if isinstance(my_val, int):
        return mesa_int(my_val, first_row=first_row)
    elif isinstance(my_val, str):
        return mesa_string(my_val, first_row=first_row)
    else:
        return mesa_float(my_val)


def get_header_rows(my_header_dict: dict[str, typing.Any]
                    ) -> list[str]:
    """Function generating the header rows for the MESA mock profile file.

    Parameters
    ----------
    my_header_dict : dict[str, typing.Any]
        Contains the necessary header information.
        
    Returns
    -------
    list
        Contains the first four (header) rows of the MESA profile file.
    """
    # get the number of header information columns
    nr_header_columns = len(my_header_dict)
    # create the string for the first row (column numbers)
    first_row = mesa_int(1, first_row=True)
    for i in range(2, nr_header_columns + 1):
        first_row += mesa_int(i)
    # create the string for the second row (column headers)
    my_head_iter = iter(my_header_dict)
    second_row = mesa_string(next(my_head_iter),
                             first_row=True)
    for k in my_head_iter:
        second_row += mesa_string(k)
    # create the string for the third row (column header values)
    my_val_iter = iter(my_header_dict.values())
    third_row = mesa_int(next(my_val_iter), first_row=True)
    for k in my_val_iter:
        third_row += handle_value(k)
    # generate the fourth, empty row
    fourth_row = mesa_string('', first_row=True)
    for _ in range(2, nr_header_columns + 1):
        fourth_row += mesa_string('')
    # return the row data
    return [first_row, second_row, third_row, fourth_row]


# create the header information
def create_header_mesa() -> tuple[str, int]:
    """Creates the header for the MESA profile mock file.
        
    Returns
    -------
    str
        The header string in MESA format.
    int
        The number of zones in the MESA profile file.
    """
    # my header information dictionary
    header_info = {
        'model_number': 5157, 'num_zones': 5, 'initial_mass': 1.0,
        'initial_z': 0.02, 'star_age': 2.0e7, 'time_step': 0.1e5,
        'Teff': 1250.0e1, 'photosphere_L': 2.3e3,
        'photosphere_r': 6.5, 'center_eta': -4.95,
        'center_h1': 9e-2, 'center_he3': 7e-10, 'center_he4': 9e-1,
        'center_c12': 1e-4, 'center_n14': 8e-3, 'center_o16': 3e-4,
        'center_ne20': 1.5e-3, 'star_mass': 6.0, 'star_mdot': -1.5e-10,
        'star_mass_h1': 3.3, 'star_mass_he3': 4.e-4,
        'star_mass_he4': 2.6, 'star_mass_c12': 5.4e-3,
        'star_mass_n14': 2e-2, 'star_mass_o16': 2.8e-2,
        'star_mass_ne20': 1.e-2, 'he_core_mass': 0.0,
        'c_core_mass': 0.0, 'o_core_mass': 0.0, 'si_core_mass': 0.0,
        'fe_core_mass': 0.0, 'neutron_rich_core_mass': 0.0,
        'tau10_mass': 6.0, 'tau10_radius': 6.5, 'tau100_mass': 6.0,
        'tau100_radius': 6.5, 'dynamic_time': 7.e4, 'kh_timescale': 6.e4,
        'nuc_timescale': 2.6e7, 'power_nuc_burn': 2.3e3,
        'power_h_burn': 2.3e3, 'power_he_burn': 9.e-28,
        'power_neu': 1.6e2, 'burn_min1': 5.e1, 'burn_min2': 1.e3,
        'time_seconds': 1.9e15, 'version_number': "15140",
        'compiler': "compiler", 'build': "10.2.0",
        'MESA_SDK_version': "x86_64-linux-21.4.1",
        'math_backend': "CRMATH", 'date': "20230119"
        }
    # generate the string rows based on the information dictionary
    my_row_list = get_header_rows(
        my_header_dict=header_info
        )
    # join into one big string and return
    return "\n".join(my_row_list), header_info['num_zones']


def get_body_rows(my_body_dict: dict[str, typing.Any],
                  num_zones: int) -> list[str]:
    """Function generating the body rows for the MESA mock profile file.

    Parameters
    ----------
    my_body_dict : dict[str, typing.Any]
        Contains the necessary body information.
    num_zones : int
        The number of zones in the MESA profile file.
        
    Returns
    -------
    list
        Contains the body rows of the MESA profile file.
    """
    # get the number of body information columns
    nr_body_columns = len(my_body_dict)
    # make a range parameter
    range_body = range(2, nr_body_columns + 1)
    # create the string for the first row (column numbers for body)
    first_row = mesa_int(1, first_row=True)
    for i in range_body:
        first_row += mesa_int(i)
    # create the string for the second row (column headers for body)
    my_body_iter = iter(my_body_dict)
    second_row = mesa_string(next(my_body_iter),
                             first_row=True)
    for k in my_body_iter:
        second_row += mesa_string(k)
    # create the string for the remaining rows (column (body) values)
    body_rows = ""
    list_vals = list(my_body_dict.values())
    for _z in range(num_zones):
        for _i in range(nr_body_columns):
            my_val = list_vals[_i][_z]
            if _i == 0:
                body_rows += handle_value(my_val, first_row=True)
            else:
                body_rows += handle_value(my_val)
        else:
            if _z != num_zones - 1:
                body_rows += '\n'
    # return the row data
    return [first_row, second_row, body_rows]


# create data array body for the MESA profile mock file
def create_body_mesa(nz: int) -> str:
    """Creates the body (data arrays) for the MESA profile mock file.
    
    Parameters
    ----------
    nz : int
        The number of zones in the MESA profile file.
        
    Returns
    -------
    str
        The body (data array) string in MESA format.
    """
    # my body information dictionary
    body_info = {
        'zone': np.linspace(1, nz, num=nz, dtype=np.int32),
        'logT': np.ones((nz,), dtype=np.float64),
        'logRho': np.ones((nz,), dtype=np.float64),
        'luminosity': np.ones((nz,), dtype=np.float64),
        'velocity':np.ones((nz,), dtype=np.float64),
        'entropy': np.ones((nz,), dtype=np.float64),
        'conv_mixing_type': 7 * np.ones((nz,), dtype=np.int32),
        'csound': np.ones((nz,), dtype=np.float64),
        'mu': np.ones((nz,), dtype=np.float64),
        'q': np.ones((nz,), dtype=np.float64),
        'radius': np.ones((nz,), dtype=np.float64),
        'tau': np.ones((nz,), dtype=np.float64),
        'pressure': np.ones((nz,), dtype=np.float64),
        'grada': np.ones((nz,), dtype=np.float64),
        'gradT': np.ones((nz,), dtype=np.float64),
        'gradr': np.ones((nz,), dtype=np.float64),
        'opacity': np.ones((nz,), dtype=np.float64),
        'eps_nuc': np.ones((nz,), dtype=np.float64),
        'non_nuc_neu': np.ones((nz,), dtype=np.float64),
        'mlt_mixing_length': np.ones((nz,), dtype=np.float64),
        'log_D_mix': np.ones((nz,), dtype=np.float64),
        'log_conv_vel': np.ones((nz,), dtype=np.float64),
        'log_D_conv': np.ones((nz,), dtype=np.float64),
        'log_D_ovr': np.ones((nz,), dtype=np.float64),
        'log_D_minimum': np.ones((nz,), dtype=np.float64),
        'pressure_scale_height': np.ones((nz,), dtype=np.float64),
        'mass': np.ones((nz,), dtype=np.float64),
        'mmid': np.ones((nz,), dtype=np.float64),
        'grav': np.ones((nz,), dtype=np.float64),
        'x': np.ones((nz,), dtype=np.float64),
        'y': np.ones((nz,), dtype=np.float64),
        'z': np.ones((nz,), dtype=np.float64),
        'h1': np.ones((nz,), dtype=np.float64),
        'h2': np.ones((nz,), dtype=np.float64),
        'he3': np.ones((nz,), dtype=np.float64),
        'he4': np.ones((nz,), dtype=np.float64),
        'c12': np.ones((nz,), dtype=np.float64),
        'c13': np.ones((nz,), dtype=np.float64),
        'n14': np.ones((nz,), dtype=np.float64),
        'o16': np.ones((nz,), dtype=np.float64),
        'o17': np.ones((nz,), dtype=np.float64),
        'ne20': np.ones((nz,), dtype=np.float64),
        'ne22': np.ones((nz,), dtype=np.float64),
        'mg24': np.ones((nz,), dtype=np.float64),
        'al27': np.ones((nz,), dtype=np.float64),
        'si28': np.ones((nz,), dtype=np.float64),
        's32': np.ones((nz,), dtype=np.float64),
        'ar36': np.ones((nz,), dtype=np.float64),
        'brunt_N2': np.ones((nz,), dtype=np.float64),
        'brunt_N2_structure_term': np.ones((nz,), dtype=np.float64),
        'brunt_N2_composition_term': np.ones((nz,), dtype=np.float64),
        'lamb_S': np.ones((nz,), dtype=np.float64),
        'lamb_S2': np.ones((nz,), dtype=np.float64),
        'hyeq_lhs': np.ones((nz,), dtype=np.float64),
        'hyeq_rhs': np.ones((nz,), dtype=np.float64),
        'gamma_1_adDer': np.ones((nz,), dtype=np.float64)
        }
    # obtain the body row list
    body_row_list = get_body_rows(my_body_dict=body_info,
                                  num_zones=nz)
    # join into one big string and return
    return "\n".join(body_row_list)
    
    
# store the expected output for the MESA read file
class ExpectedMESAProfileData:
    """Contains the expected MESA Profile file data."""
    
    # attribute and type declarations
    # - body arrays
    gamma_1_adDer: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    h1: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    h2: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    he3: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    he4: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    c12: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    c13: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    n14: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    o16: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    o17: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    ne20: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    ne22: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    mg24: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    al27: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    si28: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    s32: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    ar36: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    brunt_N2: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    brunt_N2_structure_term: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    brunt_N2_composition_term: np.ndarray = \
        (np.ones((5,), dtype=np.float64), np.ndarray)
    lamb_S: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    lamb_S2: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    hyeq_lhs: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    hyeq_rhs: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    x: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    y: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    z: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    mlt_mixing_length: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    log_D_mix: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    log_D_conv: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    log_D_ovr: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    log_D_minimum: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    conv_mixing_type: np.ndarray = (7 * np.ones((5,), dtype=np.float64), np.ndarray)
    zone: np.ndarray = (np.linspace(1., 5., num=5, dtype=np.float64), np.ndarray)
    logT: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    logRho: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    luminosity: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    velocity: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    entropy: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    csound: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    mu: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    q: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    radius: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    tau: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    pressure: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    opacity: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    eps_nuc: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    non_nuc_neu: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    log_conv_vel: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    mass: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    mmid: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    grada: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    gradT: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    gradr: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    pressure_scale_height: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    grav: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    # - header information
    model_number: int = (5157, int)
    num_zones: int = (5, int)
    initial_mass: float = (1.0, float)
    initial_z: float = (0.02, float)
    star_age: float = (2.0e7, float)
    time_step: float = (0.1e5, float)
    Teff: float = (1250.0e1, float)
    photosphere_L: float = (2.3e3, float)
    photosphere_r: float = (6.5, float)
    center_eta: float = (-4.95, float)
    center_h1: float = (9e-2, float)
    center_he3: float = (7e-10, float)
    center_he4: float = (9e-1, float)
    center_c12: float = (1e-4, float)
    center_n14: float = (8e-3, float)
    center_o16: float = (3e-4, float)
    center_ne20: float = (1.5e-3, float)
    star_mass_h1: float = (3.3, float)
    star_mass_he3: float = (4.e-4, float)
    star_mass_he4: float = (2.6, float)
    star_mass_c12: float = (5.4e-3, float)
    star_mass_n14: float = (2e-2, float)
    star_mass_o16: float = (2.8e-2, float)
    star_mass_ne20: float = (1.e-2, float)
    he_core_mass: float = (0.0, float)
    c_core_mass: float = (0.0, float)
    o_core_mass: float = (0.0, float)
    si_core_mass: float = (0.0, float)
    fe_core_mass: float = (0.0, float)
    neutron_rich_core_mass: float = (0.0, float)
    tau10_mass: float = (6.0, float)
    tau10_radius: float = (6.5, float)
    tau100_mass: float = (6.0, float)
    tau100_radius: float = (6.5, float)
    dynamic_time: float = (7.e4, float)
    kh_timescale: float = (6.e4, float)
    nuc_timescale: float = (2.6e7, float)
    power_nuc_burn: float = (2.3e3, float)
    power_h_burn: float = (2.3e3, float)
    power_he_burn: float = (9.e-28, float)
    power_neu: float = (1.6e2, float)
    burn_min1: float = (5.e1, float)
    burn_min2: float = (1.e3, float)
    time_seconds: float = (1.9e15, float)
    version_number: str = ('15140', str)
    compiler: str = ('compiler', str)
    build: str = ('10.2.0', str)
    MESA_SDK_version: str = ('x86_64-linux-21.4.1', str)
    date: str = ('20230119', str)
    
    # class method used to obtain all attributes from this class
    # in a dictionary format
    @classmethod
    def get_dict(cls) -> dict[str, typing.Any]:
        """Retrieves all attributes of this class in a dictionary format.
        
        Returns
        -------
        dict[str, typing.Any]
            The dictionary containing all attributes of this class.
        """
        return {
            a: getattr(cls, a)
            for a in vars(cls) if ('__' not in a) and ('get_dict' != a)
        }
