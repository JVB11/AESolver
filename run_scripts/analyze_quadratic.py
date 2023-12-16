"""Python run script used to analyze the (stationary) solutions to the 
quadratic amplitude equations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from aerunners import perform_complete_analysis_run
from log_formatter import set_up_root_logger


# set up root logger
logger = set_up_root_logger()


if __name__ == "__main__":
    # perform the requested analyses of the saved data
    perform_complete_analysis_run(logger=logger)
