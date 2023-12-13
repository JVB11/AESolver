Module log_formatter
====================
Initialization file for the python module that contains functions used to set up the root logger, so that it logs messages in a nice format.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>

Sub-modules
-----------
* log_formatter.log_formatter

Functions
---------

    
`adjust_root_logger_level(my_root_logger: logging.RootLogger, logger_info_dict: dict)`
:   Function used to adjust the root logger level based on
    specific arguments in the logger information dictionary.
    
    Parameters
    ----------
    my_root_logger : logging.RootLogger
        The root logger object.
    logger_info_dict : dict
        Contains the 'debug' and 'verbose' keys that define whether
        logger level needs to be adjusted.
    
    Returns
    -------
    my_root_logger : logging.RootLogger
        The rootlogger object.

    
`set_up_root_logger(my_level=20)`
:   Function that sets up the root logger.
    
    Parameters
    ----------
    my_level : int, optional
        The level of the root logger. Default: logging.INFO
    
    Returns
    -------
    logging.RootLogger
        The rootlogger object.