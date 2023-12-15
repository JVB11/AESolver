"""Python enumeration module containing information that maps to the path on the stored HDF5 file output.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from dataclasses import dataclass

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypedDict, Self

    class AttrMap(TypedDict):
        k: tuple[str, str]
        l: tuple[str, str]  # noqa: E741
        m: tuple[str, str]

    class ArrMap(TypedDict):
        surfs: tuple[str, str]
        checks: tuple[str, str]
        driv_damp_rates: tuple[str, str]
        theoretical_amps: tuple[str, str]
        thresh_amp: tuple[str, str]
        rad_ord: tuple[str, str]
        corot_mode_omegas: tuple[str, str]
        corot_mode_freqs: tuple[str, str]
        inert_mode_omegas: tuple[str, str]
        inert_mode_freqs: tuple[str, str]
        inert_dimless_mode_freqs: tuple[str, str]
        kappa: tuple[str, str]
        kappa_norm: tuple[str, str]
        stationary_relative_luminosity_phase: tuple[str, str]


@dataclass(
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=True,
    match_args=True,
    kw_only=False,
    slots=False,
    weakref_slot=False,
)
class HDF5MappingsAttributes:
    """Class that holds mappings to attributes."""

    k: tuple[str, str] = ('quantum_numbers', 'k')
    l: tuple[str, str] = ('quantum_numbers', 'l')  # noqa: E741
    m: tuple[str, str] = ('quantum_numbers', 'm')

    def get_dict(self: 'Self') -> 'AttrMap':
        return vars(self)


@dataclass(
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=True,
    match_args=True,
    kw_only=False,
    slots=False,
    weakref_slot=False,
)
class HDF5MappingsArrays:
    """Class that holds mappings to arrays."""

    surfs: tuple[str, str] = (
        'coupling/stationary_three_mode/surface_flux_variations',
        '',
    )
    checks: tuple[str, str] = ('checks/pre_checks', '')
    driv_damp_rates: tuple[str, str] = ('linear_driving/driving_rates', '')
    theoretical_amps: tuple[str, str] = (
        'coupling/stationary_three_mode/equilibrium_theoretical_amplitudes',
        '',
    )
    thresh_amp: tuple[str, str] = (
        'coupling/stationary_three_mode/critical_parent_amplitudes',
        '',
    )
    rad_ord: tuple[str, str] = ('quantum_numbers/n', '')
    corot_mode_omegas: tuple[str, str] = (
        'mode_frequency_info/corot_mode_omegas',
        '',
    )
    corot_mode_freqs: tuple[str, str] = (
        'mode_frequency_info/corot_mode_freqs',
        '',
    )
    inert_mode_omegas: tuple[str, str] = (
        'mode_frequency_info/inert_mode_omegas',
        '',
    )
    inert_mode_freqs: tuple[str, str] = (
        'mode_frequency_info/inert_mode_freqs',
        '',
    )
    inert_dimless_mode_freqs: tuple[str, str] = (
        'mode_frequency_info/inert_dimless_mode_omegas',
        '',
    )
    kappa: tuple[str, str] = ('coupling/generic/|kappa|', '')
    kappa_norm: tuple[str, str] = ('coupling/generic/|kappa_norm|', '')
    stationary_relative_luminosity_phase: tuple[str, str] = (
        'coupling/stationary_three_mode/stationary_relative_luminosity_phases',
        '',
    )

    def get_dict(self: 'Self') -> 'ArrMap':
        return vars(self)
