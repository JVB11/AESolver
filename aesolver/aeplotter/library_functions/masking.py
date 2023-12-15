"""Plotting module containing functions used for masking of data arrays that will be used to generate plots.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np


# SPECIFIC SIMPLE CONDITION MASKS


def _get_stability_check_mask(data_dict: dict) -> np.ndarray:
    """Retrieves the stability check mask based on stored checks data.

    Parameters
    ----------
    data_dict : dict
        Contains the checks data.

    Returns
    -------
    np.ndarray
        Stability check mask (True if stable, False if not).
    """
    return data_dict['checks'][:, -3:].prod(axis=1, dtype=np.bool_)


def _get_ae_validity_check_mask(data_dict: dict) -> np.ndarray:
    """Retrieves the AE validity check mask.

    Parameters
    ----------
    data_dict : dict
        Contains the data necessary to perform the check.

    Returns
    -------
    np.ndarray
        AE validity mask.
    """
    return np.abs(data_dict['linear_omega_diff_to_min_omega']) <= 0.1


def _get_isolation_mask(lowest_thresholds: np.ndarray) -> np.ndarray:
    """Generates the mode triad isolation mask.

    Parameters
    ----------
    lowest_thresholds : np.ndarray
        Contains the lowest thresholds for the different modes in the triad.

    Returns
    -------
    np.ndarray
        Verifies if the mode threshold amplitudes are the same for each of the modes (i.e. a necessary but not sufficient condition for mode triad isolation).
    """
    # make individual comparisons at floating point precision
    pd1 = np.isclose(
        lowest_thresholds[:, 0], lowest_thresholds[:, 1], rtol=0, atol=1e-14
    )
    pd2 = np.isclose(
        lowest_thresholds[:, 0], lowest_thresholds[:, 2], rtol=0, atol=1e-14
    )
    # return the isolation mask
    return pd1 & pd2


def _threshold_det_mask(
    data_dict: dict, max_threshold_detuning: float
) -> np.ndarray:
    """Generates mask that ensures that the linear frequency detuning is not larger than a specific supplied maximal threshold value.

    Parameters
    ----------
    data_dict : dict
        Contains the necessary data with which comparisons need to be made.
    max_threshold_detuning : float
        Maximal value of the linear frequency detuning of the mode triad. This is given in terms of the fraction of the smallest mode frequency of the mode triad. For example, if a mode triad consists of modes with frequencies 0.3, 0.2, and 0.99 cycles/day, than the mode detuning shall be required to be less than 'max_threshold_detuning'*(0.99 cycles/day). Typical values are 0.1, to ensure that the detuning is at least 10 times smaller than the smallest frequency, thereby ensuring that the timescales considered in the amplitude equations are slower than the individual mode timescales.

    Returns
    -------
    thresh_detuning_ok_mask : np.ndarray
        Maximal threshold detuning mask.
    """
    # get mask for threshold detunings
    thresh_detuning_ok_mask = (
        np.abs(data_dict['linear_omega_diff_to_min_omega'])
        <= max_threshold_detuning
    )
    # return the threshold mask
    return thresh_detuning_ok_mask


# ISOLATION MASK AND CORRESPONDING UTILITY FUNCTIONS


def _get_isolation_mask_lowest_thresh_amp(data_dict):
    # get necessary info to generate isolation mask
    _my_thresh = data_dict['thresh_surf_ppm']
    # get the values mask
    _is_valued_and_parametric = ~np.isnan(_my_thresh)
    _valued_thresh = _my_thresh[_is_valued_and_parametric]
    # get the relevant radial orders
    _valued_rad_order = data_dict['rad_ord'][_is_valued_and_parametric]
    # PER UNIQUE N of a PARENT OR DAUGHTER, find the LOWEST THRESH SURF LUMINOSITY, and return the unique relevant radial orders
    (
        _lowest_thresh_parent_map,
        _unique_n_parent,
    ) = _get_lowest_thresh_amp_map_and_unique_values(
        rad_orders=_valued_rad_order, thresholds=_valued_thresh, idx=0
    )
    (
        _lowest_thresh_daughter_1_map,
        _unique_n_daughters_1,
    ) = _get_lowest_thresh_amp_map_and_unique_values(
        rad_orders=_valued_rad_order, thresholds=_valued_thresh, idx=1
    )
    (
        _lowest_thresh_daughter_2_map,
        _unique_n_daughters_2,
    ) = _get_lowest_thresh_amp_map_and_unique_values(
        rad_orders=_valued_rad_order, thresholds=_valued_thresh, idx=2
    )
    # check for the lowest threshold amplitudes among the daughters
    _lowest_thresh_daughters_map = _lowest_thresh_daughter_1_map.copy()
    for _k, _lt_d2 in _lowest_thresh_daughter_2_map.items():
        try:
            # add to dict
            if _lowest_thresh_daughter_1_map[_k] > _lt_d2:
                _lowest_thresh_daughters_map[_k] = _lt_d2
        except KeyError:
            # add to dict
            _lowest_thresh_daughters_map[_k] = _lt_d2
    # construct numpy array containing lowest thresh surf
    _lowest_thresh_surf = np.full_like(
        _valued_rad_order, np.nan, dtype=np.float64
    )
    _get_lowest_thresh_surfs(
        rad_ords=_valued_rad_order,
        unique_rad_ords=_unique_n_parent,
        surf_map=_lowest_thresh_parent_map,
        lowest_thresholds=_lowest_thresh_surf,
        idx=0,
    )
    _get_lowest_thresh_surfs(
        rad_ords=_valued_rad_order,
        unique_rad_ords=_unique_n_daughters_1,
        surf_map=_lowest_thresh_daughters_map,
        lowest_thresholds=_lowest_thresh_surf,
        idx=1,
    )
    _get_lowest_thresh_surfs(
        rad_ords=_valued_rad_order,
        unique_rad_ords=_unique_n_daughters_2,
        surf_map=_lowest_thresh_daughters_map,
        lowest_thresholds=_lowest_thresh_surf,
        idx=2,
    )
    # look for mode triads that have the same surface threshold luminosities: THESE ARE ISOLATED MODE TRIADS (IF OTHER TRIADS HAVE HIGHER THRESHOLDS)!
    _all_same = _get_isolation_mask(lowest_thresholds=_lowest_thresh_surf)
    # get original-size mask
    og_size_mask = np.zeros_like(_is_valued_and_parametric, dtype=np.bool_)
    og_size_mask[_is_valued_and_parametric] = _all_same
    # ALSO VERIFY WHETHER THE SECOND LOWEST THRESHOLD AMPLITUDE IS HIGHER THAN THE EQUILIBRIUM AMPLITUDE OF THE TRIAD WITH LOWEST THRESHOLD AMPLITUDE, IF REQUESTED
    second_lowest_ok = (
        compare_second_lowest_thresholds_with_lowest_threshold_equilibrium_amp(
            is_valued_and_parametric=_is_valued_and_parametric,
            unique_parent_rad_orders=_unique_n_parent,
            data_dict=data_dict,
        )
    )
    # return isolation mask
    return og_size_mask & second_lowest_ok


def _get_lowest_thresh_amp_map_and_unique_values(
    rad_orders: np.ndarray, thresholds: np.ndarray, idx: list | np.ndarray
) -> dict:
    """Retrieves a map of the lowest threshold amplitudes to the corresponding (unique) radial orders.

    Parameters
    ----------
    rad_orders : np.ndarray
        The radial orders under investigation.
    thresholds : np.ndarray
        The threshold amplitudes under investigation.
    idx : list | np.ndarray
        Indices used to enforce earlier defined masks.

    Returns
    -------
    dict
        Mapping of the (unique) radial orders to the corresponding lowest threshold amplitudes.
    """
    # get relevant radial order
    _my_relevant_rad_order = rad_orders[:, idx]
    # get unique radial order
    my_unique_relevant_rad_orders = np.unique(_my_relevant_rad_order)
    # return the threshold amplitude map
    return {
        _n_relevant: thresholds[
            np.isin(_my_relevant_rad_order, _n_relevant)
        ].min()
        for _n_relevant in my_unique_relevant_rad_orders
    }, my_unique_relevant_rad_orders


def _get_lowest_thresh_surfs(
    rad_ords: np.ndarray,
    unique_rad_ords: np.ndarray,
    surf_map: dict,
    lowest_thresholds: np.ndarray,
    idx: list | np.ndarray,
) -> None:
    """Stores the lowest threshold surface luminosity fluctuations.

    Parameters
    ----------
    rad_ords : np.ndarray
        The radial orders under investigation.
    unique_rad_ords : np.ndarray
        The unique radial orders under investigation.
    surf_map : dict
        Mapping of the unique radial orders to the corresponding lowest threshold amplitudes.
    lowest_thresholds : np.ndarray
        Array storing these lowest thresholds.
    idx : list | np.ndarray
        Indices used to enforce earlier defined masks.
    """
    my_rad_ords = rad_ords[:, idx]
    for n_unique in unique_rad_ords:
        lowest_thresholds[np.isin(my_rad_ords, n_unique), idx] = surf_map[
            n_unique
        ]


def compare_second_lowest_thresholds_with_lowest_threshold_equilibrium_amp(
    is_valued_and_parametric, unique_parent_rad_orders, data_dict
):
    # retrieve a dummy variable storing the parent radial orders
    n_parents = data_dict['rad_ord'][is_valued_and_parametric, 0]
    # initialize the mask array
    mask_arr = np.zeros_like(n_parents, dtype=np.bool_)
    # retrieve a dummy variable representing the equilibrium amplitudes of the parents, in PPM
    eq_amps = data_dict['surfs'][is_valued_and_parametric, 0] * 1.0e6
    threshs = data_dict['thresh_surf_ppm'][is_valued_and_parametric]
    # loop over the unique parent radial orders to generate the mask
    for _n_p in unique_parent_rad_orders:
        # get mask for parent radial orders
        _mask_parent = n_parents == _n_p
        # get equilibrium and threshold amplitudes
        _eqs = eq_amps[_mask_parent]
        _threshs = threshs[_mask_parent]
        # determine the second smallest threshold amplitude
        _arg_part = np.argpartition(_threshs, 1)
        _second_lowest_thresh = _threshs[_arg_part[1]]
        _eq_amp_lowest_thresh = _eqs[_arg_part[0]]
        # fill up the mask array
        mask_arr[_mask_parent] = _second_lowest_thresh > _eq_amp_lowest_thresh
    # update that mask so that it represents the original size
    og_size_mask_arr = np.zeros_like(is_valued_and_parametric, dtype=np.bool_)
    og_size_mask_arr[is_valued_and_parametric] = mask_arr
    # print whether there were any parent radial orders that were masked due to this threshold comparison
    if len(invalid_parents := np.unique(n_parents[~mask_arr])) > 0:
        print(
            f'The following parent radial orders were not considered for isolated couplings: {invalid_parents}. The reason for this neglect: the second lowest threshold amplitude is lower than the saturation/equilibrium amplitude of the parent of the parametric resonant coupling triad with the lowest threshold amplitude.'
        )
    # return comparison mask
    return og_size_mask_arr


# CUSTOMIZED MASKING FUNCTIONS


def _get_threshold_lum_mask(
    data_dict: dict, minimum_threshold_lum: float | str
) -> np.ndarray:
    """Retrieve the mask based on a supplied minimum threshold luminosity.

    Parameters
    ----------
    data_dict : dict
        Contains the data with which comparisons need to be made.
    minimum_threshold_lum : float | str
        Supplied minimum threshold luminosity.

    Returns
    -------
    np.ndarray
        Threshold luminosity mask.
    """
    # get mask for threshold amplitudes
    _thresh_lum_too_low_or_nan_mask = (
        data_dict['thresh_surf_ppm'] < float(minimum_threshold_lum)
    ) | np.isnan(data_dict['thresh_surf_ppm'])
    # get the radial orders associated with triads that have threshold luminosity variations smaller than 'minimum_threshold_lum'
    _radial_orders_too_low_thresh_lum = data_dict['rad_ord'][
        _thresh_lum_too_low_or_nan_mask
    ]
    # get unique values of the radial orders of modes that partake in mode triads whose threshold amplitude is below 'minimum_threshold_lum' and use those unique values to get per-mode masks
    parent_mask = _get_unique_value_mask(
        radial_orders_to_be_ignored=_radial_orders_too_low_thresh_lum,
        data_dict=data_dict,
        idx=0,
    )
    daughter_1_mask = _get_unique_value_mask(
        radial_orders_to_be_ignored=_radial_orders_too_low_thresh_lum,
        data_dict=data_dict,
        idx=1,
    )
    daughter_2_mask = _get_unique_value_mask(
        radial_orders_to_be_ignored=_radial_orders_too_low_thresh_lum,
        data_dict=data_dict,
        idx=2,
    )
    # construct the threshold luminosity mask that selects mode triads that have modes with radial orders occurring in mode triads whose threshold amplitude is below 'minimum_threshold_lum'
    thresh_lum_mask = parent_mask | daughter_1_mask | daughter_2_mask
    # return the inverse of the above mask to select 'independent' mode triads
    return ~thresh_lum_mask


def _get_unique_value_mask(
    radial_orders_to_be_ignored: list | np.ndarray,
    data_dict: dict,
    idx: int | list[int] | np.ndarray[int],
) -> np.ndarray:
    """Retrieves a unique value mask that is used to indicate mode triad data entries that do not fulfill the threshold luminosity criterion.

    Parameters
    ----------
    radial_orders_to_be_ignored : list | np.ndarray
        Contains the specific radial orders that should be ignored in the creation of the mask.
    data_dict : dict
        Contains the radial orders that need to be compared against.
    idx : int | list[int] | np.ndarray[int]
        Index or list/array of indices used to mask the arrays stored in the data dict, before comparison (i.e. used to retrieve a certain, masked subset of these data arrays!).

    Returns
    -------
    np.ndarray
        Mask that retrieves the data entries containing unique values of radial orders.
    """
    unique_radial_orders_to_be_ignored = np.unique(
        radial_orders_to_be_ignored[:, idx]
    )
    return np.isin(
        data_dict['rad_ord'][:, idx], unique_radial_orders_to_be_ignored
    )


# GENERIC PART OF RUN FUNCTIONS


def _get_hyper_ae_isolation_masks(
    data_dict: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get masks for properties that always need to be fulfilled
    hyper_mask = _get_stability_check_mask(data_dict=data_dict)
    ae_mask = _get_ae_validity_check_mask(data_dict=data_dict)
    # get isolation mask
    isolation_mask = _get_isolation_mask_lowest_thresh_amp(data_dict=data_dict)
    # return masks
    return hyper_mask, ae_mask, isolation_mask


# SPECIFIC RUN FUNCTIONS


def theoretical_plotting_masks_overview(
    data_dict: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs masking for the overview plots.

    Parameters
    ----------
    data_dict : dict
        Contains the necessary data for constructing the masks.

    Returns
    -------
    stable_mask : np.ndarray
        Mask denoting the stable solutions (not AE-valid and not isolated).
    ae_stable_mask : np.ndarray
        Mask denoting the stable AE-valid solutions (not isolated).
    stable_valid_mask : np.ndarray
        Mask denoting the isolated solutions (stable, AE-valid and isolated).
    """
    # get the harmonic removal mask
    harmonic_removal_mask = ~data_dict['harmonic_mask']
    # compute individual masks for the three categories of points considered:
    # 1) STABLE ONLY
    # 2) STABLE + AE-VALID
    # 3) STABLE + VALID (I.E. ISOLATED)
    # -- get individual masks
    (
        _stab_mask_all,
        _ae_validity_all,
        _isolation_all,
    ) = _get_hyper_ae_isolation_masks(data_dict=data_dict)
    # -- create the categorical masks
    stable_mask = _stab_mask_all & harmonic_removal_mask
    ae_stable_mask = _stab_mask_all & _ae_validity_all & harmonic_removal_mask
    stable_valid_mask = (
        _stab_mask_all
        & _ae_validity_all
        & _isolation_all
        & harmonic_removal_mask
    )
    # -- exclude overlap for the less stringent categories
    stable_mask = (
        stable_mask
        & ~ae_stable_mask
        & ~stable_valid_mask
        & harmonic_removal_mask
    )
    ae_stable_mask = ae_stable_mask & ~stable_valid_mask & harmonic_removal_mask
    # return the masks
    return stable_mask, ae_stable_mask, stable_valid_mask


def isolated_non_isolated_stable_valid_masking(
    data_dict: dict,
    minimum_threshold_lum: float | None,
    max_threshold_detuning: float | None,
) -> np.ndarray:
    """Performs masking for the resonance sharpness plot.

    Parameters
    ----------
    data_dict : dict
        Contains the necessary data for constructing the mask.
    hyperbolic : bool
        If True, enforce hyperbolicity.

    Returns
    -------
    np.ndarray
        Generic mask based on the different enforced conditions.

    Raises
    ------
    ValueError
        Raised when trying to compute non-hyperbolic stationary solutions, as we cannot ascertain that those solutions are stable to small perturbations around the stationary points!
    """
    # get generic masks
    _hyper_mask, _ae_mask, _isolation_mask = _get_hyper_ae_isolation_masks(
        data_dict=data_dict
    )
    # now get masks with and without isolation
    with_isolation = _hyper_mask & _ae_mask & _isolation_mask
    without_isolation = _hyper_mask & _ae_mask & ~_isolation_mask
    # additional masks
    if (minimum_threshold_lum is not None) and (
        max_threshold_detuning is not None
    ):
        _threshold_lum_mask = _get_threshold_lum_mask(
            data_dict=data_dict, minimum_threshold_lum=minimum_threshold_lum
        )
        _threshold_detuning_mask = _threshold_det_mask(
            data_dict=data_dict, max_threshold_detuning=max_threshold_detuning
        )
        # return both masks
        return (
            with_isolation & _threshold_lum_mask & _threshold_detuning_mask
        ), (without_isolation & _threshold_lum_mask & _threshold_detuning_mask)
    elif minimum_threshold_lum is not None:
        _threshold_lum_mask = _get_threshold_lum_mask(
            data_dict=data_dict, minimum_threshold_lum=minimum_threshold_lum
        )
        return (with_isolation & _threshold_lum_mask), (
            without_isolation & _threshold_lum_mask
        )
    elif max_threshold_detuning is not None:
        _threshold_detuning_mask = _threshold_det_mask(
            data_dict=data_dict, max_threshold_detuning=max_threshold_detuning
        )
        return (with_isolation & _threshold_detuning_mask), (
            without_isolation & _threshold_detuning_mask
        )
    else:
        return with_isolation, without_isolation
