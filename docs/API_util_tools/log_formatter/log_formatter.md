---
layout: api_module_page
title: log-formatting functions API reference
permalink: /overview_API/API_util_tools/log_formatter/log_formatter.html
---

# log_formatter.log_formatter module

Python module containing functions used to set up the root logger to format log messages in a nice way.

{% include button_api_module.html referenced_path="/tree/main/util-tools/log_formatter/log_formatter.py" %}

## Functions

`adjust_root_logger_level(my_root_logger: logging.RootLogger, logger_info_dict: dict)`
:   Function used to adjust the root logger level based on specific arguments in the logger information dictionary.

    Parameters
    ----------
    my_root_logger : logging.RootLogger
        The root logger object.
    logger_info_dict : dict
        Contains the 'debug' and 'verbose' keys that define whether logger level needs to be adjusted.
    
    Returns
    -------
    my_root_logger : logging.RootLogger
        The rootlogger object.

`set_up_root_logger(my_level=20)`
:   Function that sets up the root logger.

    Parameters
    ----------
    my_level : int, optional
        The level of the root logger, by default 'logging.INFO'.
    
    Returns
    -------
    logging.RootLogger
        The rootlogger object.
