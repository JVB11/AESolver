'''Python module containing utility functions for the runner classes.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''


def _convert_integer_to_count_string(pos: int) -> str:
    """Generates a string representing an integer.

    Parameters
    ----------
    pos : int
        The integer that needs to be represented in a string. In this case, this integer refers to a position in the read command line input arguments.

    Returns
    -------
    str
        String representation of the passed integer 'pos'.
    """
    units = ['','one','two','three','four','five','six','seven','eight','nine']
    teens = ['','eleven','twelve','thirteen','fourteen','fifteen','sixteen', 'seventeen','eighteen','nineteen']
    tens = ['','ten','twenty','thirty','forty','fifty','sixty','seventy', 'eighty', 'ninety']
    words = []
    if pos == 0:
        words.append('zero')
    else:
        num_str = f'{pos}'
        num_str_len = len(num_str)
        groups = int((num_str_len + 2) / 3)
        x3g = groups * 3
        num_str = num_str.zfill(x3g)
        for _i in range(0, x3g, 3):
            h,t,u = int(num_str[_i]),int(num_str[_i+1]),int(num_str[_i+2])
            if h >= 1:
                words.append(units[h])
                words.append('hundred')
            if t > 1:
                words.append(tens[h])
                if u >= 1:
                    words.append(units[u])
            elif t == 1:
                if u >= 1:
                    words.append(teens[u])
                else:
                    words.append(tens[t])
            else:
                if u >= 1:
                    words.append(units[u])
    return ' '.join(words)


def check_save_file_name(sys_arguments: list[str], pos: int) -> bool:
    """Checks if the required 'save_file_name' keyword argument is passed via the command line. If it is not passed, a RuntimeError is raised.

    Parameters
    ----------
    sys_arguments : list[str]
        Contains the system arguments, read from the command line.
    pos : int
        Position of the specific argument you are checking, in your setup.

    Returns
    -------
    bool
        True if the element is present, False if not.

    Raises
    ------
    RuntimeError
        Raised when the '--save_file_name' keyword argument is not passed via the command line.
    """
    if sys_arguments[pos] != '--save_file_name':
        # this is not the case: report by raising error!
        raise RuntimeError(f'{_convert_integer_to_count_string(pos=pos + 1).capitalize()} input argument{"s were" if pos > 0 else " was"} given ({sys_arguments[1:]}), but you are not specifying the --save_file_name argument for the argumentparser.\nThis is incorrect behavior.\nPlease either specify:\n*) inlist file name, inlist file suffix and relative path to the inlist file from the repository base directory, followed by "--save_file_name xxx" (to read a custom inlist)\n*) "--save_file_name xxx" only (to use the default inlist).\nOther options have not yet been implemented.')
