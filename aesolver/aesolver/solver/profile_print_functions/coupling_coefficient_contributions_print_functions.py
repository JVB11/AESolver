"""Python module containing functions that print information about the coupling coefficient contribution profiles.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""


def print_cc_contribution_info(
    cc_contribution: float,
    adjusted_cc_contribution: float,
    masked_contribution: float | list[float],
    contribution_identifier_string: str,
) -> None:
    """Generic function used to print information related to a specific coupling coefficient contribution term.

    Parameters
    ----------
    cc_contribution : float
        Value of the contribution.
    adjusted_cc_contribution : float
        Value of the adjusted contribution.
    masked_contribution : np.ndarray
        Values of the integrated contributions for the masked zones.
    contribution_identifier_string : str
        Identifies which contribution term we are dealing with.
    """
    # print statement on values
    print(
        f'{contribution_identifier_string.capitalize()} normed coupling coefficient (eta) contribution: {cc_contribution}, after adjustment for TAR validity: {adjusted_cc_contribution}.'
    )
    # print statement on the masked contributions
    print(
        f'Contributions to the {contribution_identifier_string} normed coupling coefficient (eta) contribution due to TAR non-validity zones: {masked_contribution}. These contributions are ordered in terms of the values of r/R for which the TAR validity criterion start to become invalid: the first number for example is the contribution of the TAR invalidity region in the core region.'
    )
