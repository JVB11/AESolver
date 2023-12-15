"""Module storing data for the plotting actions used to generate the overview plot.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from dataclasses import dataclass, field

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypedDict


    class PlotOptions(TypedDict):
        c: str
        a: float
        l: str  # noqa: E741
        z: int


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
class OverviewPlotOptions:
    """Class that holds plotting kwargs for different categories in the overview plot."""

    stab: 'PlotOptions' = field(default_factory=dict)
    ae_stab: 'PlotOptions' = field(default_factory=dict)
    stab_valid: 'PlotOptions' = field(default_factory=dict)

    def __init__(self) -> None:
        object.__setattr__(
            self, 'stab', {'c': 'orange', 'a': 0.5, 'l': 'Stab.', 'z': 0}
        )
        object.__setattr__(
            self,
            'ae_stab',
            {'c': '#91bfdb', 'a': 0.8, 'l': 'Stab. + AE', 'z': 5},
        )
        object.__setattr__(
            self,
            'stab_valid',
            {'c': 'k', 'a': 1.0, 'l': 'Stab. + Val.', 'z': 10},
        )

    def get_plot_categories(self) -> list[str]:
        return [x for x in vars(self)]

    def get_dict(self) -> 'dict[str, PlotOptions]':
        return vars(self)
