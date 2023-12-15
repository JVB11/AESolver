"""Python module containing functions used to poll a user to generate coupling coefficient and
related profiles (for plotting purposes).

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import custom modules
from simple_poller import Poller

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import mode information object
    from ....mode_input_generator import InputGen


# VERIFIER FUNCTION: verifies if the selected input is OK
def _verify_user_input() -> bool:
    """Function that verifies if the user input was perceived to be ok according to the user.

    Returns
    -------
    ok_input : bool
        True if the user accepts the input. False if the user does not accept the input
        (which will prompt the user to select different input).
    """
    # generate the poller object
    my_poller = Poller(
        poll_txt='Is this combination OK, or do you want to use a different combination? (y/N)\n',
        no_txt='Not accepted.',
        yes_text='Accepted.',
        default_txt='Not accepted.',
        exit_on_no=False,
        return_bool=True,
    )
    # assess whether the user wants to compute the mode coupling coefficient profile, and return the acceptance bool
    ok_input = my_poller.boolean_poll(default_answer='Not accepted.')
    if TYPE_CHECKING:
        assert isinstance(ok_input, bool)
    return ok_input


# RUN FUNCTION: ask if the selected profile is ok or request for a different profile to be chosen among the loaded datasets.
def poll_user_profile_for_combination_number(
    nr_loaded_data: int, mode_info_object: 'InputGen'
) -> int:
    """Polls the user, requesting information necessary to retrieve the information used to compute
    the coupling coefficient profile and related profiles.

    Parameters
    ----------
    nr_loaded_data : int
        The total number of loaded data, which is used to make a selection between different data sets
        relating to different mode triads.
    mode_info_object : object
        Contains the generic mode information of the modes in the triad(s).

    Returns
    -------
    int
        The selection integer that selects a specific mode triad to be used for computing and
        plotting the coupling coefficient and related profiles.
    """
    if TYPE_CHECKING:
        my_rad_ord_combo = 0
    # initialize loop bool
    not_ok = True
    while not_ok:
        # select a radial order combo
        try:
            my_rad_ord_combo = int(
                input(
                    f'Please specify a radial order combination number between 0 and {nr_loaded_data - 1}: '
                )
            )
            print(
                f"You have selected the number {my_rad_ord_combo}, corresponding to the combination of radial orders {mode_info_object.combinations_radial_orders[my_rad_ord_combo]} for the modes with quantum numbers 'l' {mode_info_object.mode_l} and 'm' {mode_info_object.mode_m} ."
            )
            # verify whether the user wants to use this input
            ok_input = _verify_user_input()
            # update the loop variable
            not_ok = not ok_input
        except ValueError:
            print(
                f'Unknown integer input {my_rad_ord_combo}. Please give a valid integer!'
            )
        except TypeError:
            print(
                f'Unknown integer input {my_rad_ord_combo}. Please give a valid integer!'
            )
    # return the combination number/selection integer
    return my_rad_ord_combo
