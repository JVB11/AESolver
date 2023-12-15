"""Python module containing functions that are used to set up the environment to be used to save the profile figures.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import pathlib as pl

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import mode info object
    from typing import Any


# get model-dependent part of save name
def get_model_dependent_part_save_name(
    save_name: str, profile_selection_dict: 'dict[str, Any]', rad_order_tuple: tuple[int, int, int]) -> str:
    """Retrieves the model dependent part of the save name.

    Parameters
    ----------
    save_name : str
        Initial string used to generate save name (user-specified).
    mode_info_object : InputGen
        Contains necessary information about the selected triad.
    rad_ord_combo : int
        Triad combination number.

    Returns
    -------
    str
        Model-dependent part of the save name.
    """
    # construct radial order part
    radial_order_part = '_'.join([str(x) for x in rad_order_tuple])
    # construct spherical degree part
    spherical_degree_part = '_'.join([str(x) for x in profile_selection_dict['mode_l']])
    # construct azimuthal order part
    azimuthal_order_part = '_'.join([str(x) for x in profile_selection_dict['mode_m']])
    # return the model-dependent part
    return f'_{save_name}_n{radial_order_part}_l{spherical_degree_part}_m{azimuthal_order_part}'


def get_figure_output_base_path(
    base_path: str = 'profiles_isolated_mode_triads',
    mesa_specific_path: str | None = None,
    rot_percent_string: str | None = None,
) -> pl.Path:
    """Retrieves the base output path for figures.

    Parameters
    ----------
    base_path : str, optional
        The path to the base/top directory in which figures will be saved; by default 'profiles_isolated_mode_triads'.
    mesa_specific_path : str, optional
        The sub-path (starting from the figure base directory) that denotes a directory in which model-specific figure file output shall be stored. If None, no subdirectory is specified; by default None.
    rot_percent_string : str, optional
        Denotes the percentage of critical rotation used in the (GYRE) models. If None, no rotation (as a percentage of Roche critical rotation) is specified; by default None.

    Returns
    -------
    pl.Path
        Base figure output path.
    """
    # get path string
    _my_string_path = f'./figure_output/{base_path}'
    if mesa_specific_path is not None:
        _my_string_path += f'/{mesa_specific_path}'
    if rot_percent_string is not None:
        _my_string_path += f'/{rot_percent_string}'
    # get path
    my_figure_output_path = pl.Path(_my_string_path)
    # ensure that directory exists
    my_figure_output_path.mkdir(parents=True, exist_ok=True)
    # return the path
    return my_figure_output_path


def save_fig(
    figure_path: pl.Path,
    figure_base_name: str,
    figure_subdir: str,
    model_name_part: str,
    fig,
    save_as_pdf: bool = False,
) -> None:
    # create paths
    _cc_path = figure_path / f'{figure_subdir}/'
    _cc_path.mkdir(parents=True, exist_ok=True)
    # save figure
    if save_as_pdf:
        fig.savefig(
            f'{_cc_path}/{figure_base_name}{model_name_part}.pdf',
            dpi='figure',
            format='pdf',
            bbox_inches='tight',
        )
    else:
        fig.savefig(
            f'{_cc_path}/{figure_base_name}{model_name_part}.png',
            dpi='figure',
            format='png',
            bbox_inches='tight',
            transparent=True,
        )
