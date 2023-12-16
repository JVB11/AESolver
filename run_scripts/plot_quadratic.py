"""Python run script used to compute the quadratic
coupling coefficients and show the (stationary) solutions to the 
quadratic amplitude equations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from aerunners import perform_complete_plotting_run
from log_formatter import set_up_root_logger


# set up root logger
logger = set_up_root_logger()


if __name__ == "__main__":
    # generate the necessary plots for the plot run and save them
    perform_complete_plotting_run(logger=logger)
