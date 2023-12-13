---
layout: default
title: index for log_formatter module
permalink: /API_util_tools/log_formatter/index.html
---

Module log_formatter

====================

Initialization file for the python module that contains functions used to set up the root logger, so that it logs messages in a nice format.

Detailed information on the available private and publicly available functions is available in the [module API reference](API_util_tools/log_formatter/log_formatter.html).

Sub-modules
-----------

* log_formatter.log_formatter

Functions
---------

`adjust_root_logger_level(my_root_logger: logging.RootLogger, logger_info_dict: dict)`
:   Function used to adjust the root logger level based on specific arguments in the logger information dictionary.

`set_up_root_logger(my_level=20)`
:   Function that sets up the root logger.
