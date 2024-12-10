"""Python module containing functions used to set up the root logger to format log messages in a nice way.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
import logging


def set_up_root_logger(my_level: int=logging.INFO) -> logging.RootLogger:
    """Function that sets up the root logger.

    Parameters
    ----------
    my_level : int, optional
        The level of the root logger; by default: logging.INFO.

    Returns
    -------
    logging.RootLogger
        The root logger object.
    """
    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
        level=my_level,
    )
    return logging.getLogger()


def adjust_root_logger_level(
    my_root_logger: logging.RootLogger, logger_info_dict: dict
) -> logging.RootLogger:
    """Function used to adjust the root logger level based on specific arguments in the logger information dictionary.

    Parameters
    ----------
    my_root_logger : logging.RootLogger
        The root logger object.
    logger_info_dict : dict
        Contains the 'debug' and 'verbose' keys that define whether the logger level needs to be adjusted.

    Returns
    -------
    my_root_logger : logging.RootLogger
        The root logger object.
    """
    my_level = my_root_logger.getEffectiveLevel()
    if logger_info_dict['debug']:
        my_level = logging.DEBUG
    elif not logger_info_dict['verbose']:
        my_level = logging.WARNING
    my_root_logger.setLevel(my_level)
    _handler = my_root_logger.handlers[0]
    _handler.setLevel(my_level)
    # remove information from the dictionary that is not needed after adjustment
    logger_info_dict.pop('debug')
    logger_info_dict.pop('verbose')
    return my_root_logger
