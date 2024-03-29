"""Python script used to create stability-domain-visualizing figures for Van Beeck et al. (2024).

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import matplotlib
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.colors import ListedColormap

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypedDict, TypeAlias, Sequence
    from matplotlib.axes import Axes as Axes
    from matplotlib.figure import Figure as Figure
    import numpy.typing as npt

    ExtentType: TypeAlias = tuple[float, float, float, float]

    class AxesInformationDict(TypedDict):
        ax_list: list[Axes]
        plot_xlabel: list[bool]
        plot_ylabel: list[bool]
        y_left: list[bool]
        x_bottom: list[bool]
        text_colors: list[str]

    class GridInformationDict(TypedDict):
        domega_gamma1: list[float]
        theta_min: list[float]
        theta_max: list[float]
        dtheta: list[float]

    class MetaDataDict(TypedDict):
        Title: str
        Author: str
        Description: str
        Software: str

    class LoopDataDict(TypedDict):
        domega_gamma1: list[float]
        theta_min: list[float]
        theta_min: list[float]
        dtheta: list[float]
        ax_list: list[Axes]
        plot_xlabel: list[bool]
        plot_ylabel: list[bool]
        y_left: list[bool]
        x_bottom: list[bool]
        text_colors: list[str]

    # type the subplot array of axes
    # (see https://stackoverflow.com/questions/72649220/precise-type-annotating-array-numpy-ndarray-of-matplotlib-axes-from-plt-subplo)
    NDArrayOfAxes: TypeAlias = (
        'np.ndarray[Sequence[Sequence[Axes]], np.dtype[np.object_]]'
    )


# Use local TeX installation for fonts in the figure
plt.rcParams.update(
    {
        'text.usetex': True,
        'font.family': 'serif',
        'font.sans-serif': 'Times',
    }
)


# compute meshgrid and grid extent parameters
def comp_meshgrid(
    theta_min: 'float' = 1.0e-1,
    theta_max: 'float' = 100.0,
    dtheta: 'float' = 0.1,
) -> 'tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], ExtentType]':
    """Computes the meshgrid and the grid extent parameters for the generated figures.

    Parameters
    ----------
    theta_min : float, optional
        Minimum value of theta in the meshgrid, by default 1.0e-1
    theta_max : float, optional
        Maximum value of theta in the meshgrid, by default 100.0
    dtheta : float, optional
        Step (in terms of theta) between cells of the meshgrid, by default 0.1

    Returns
    -------
    X : np.ndarray[np.float64]
        X values of the meshgrid.
    Y : np.ndarray[np.float64]
        Y values of the meshgrid.
    extent : ExtentType
        Information on the extent of the values in the meshgrid.
    """
    # compute range array
    theta_range = np.arange(theta_min, theta_max, dtheta)
    # generate meshgrid
    X, Y = np.meshgrid(theta_range, theta_range)
    # store information on the extent of the meshgrid in a tuple of floats
    extent = theta_min, theta_max, theta_min, theta_max
    return X, Y, extent


# original dziembowski conditions with theta input
def original_dziembowski(
    theta2: 'npt.NDArray[np.float64]',
    theta3: 'npt.NDArray[np.float64]',
    domega_gamma1: 'float',
) -> 'npt.NDArray[np.bool_]':
    """Determines whether a stationary point of the AEs is stable, if it is hyperbolic, as claimed by Dziembowski (1982).

    Notes
    -----
    Uses the stability conditions defined in Dziembowski (1982), in which an error was likely made.

    Parameters
    ----------
    theta2 : np.ndarray[np.float64]
        X-Part of the meshgrid containing values of the dimensionless theta_2 parameter (-gamma_2 / gamma_1).
    theta3 : np.ndarray[np.float64]
        Y-Part of the meshgrid containing values of the dimensionless theta_3 parameter (-gamma_3 / gamma_1).
    domega_gamma1 : float
        Fixed value for delta omega / gamma_1 (i.e. linear frequency detuning / linear parent driving rate).

    Returns
    -------
    np.ndarray[np.bool_]
        True if stable, False if not.
    """
    # compute intermediate values
    theta23 = theta2 * theta3
    thetas = theta2 + theta3 - 1.0
    # compute the coefficients for the quartic condition
    coeffq0 = -((thetas) ** 3.0) - 2.0 * theta23
    coeffq2 = -12.0 * theta23 + thetas * (
        (theta2 - theta3) ** 2.0 + (theta2 + theta3) ** 2.0 + 2.0
    )
    coeffq4 = 3.0 * (
        -6.0 * theta23
        + thetas
        * ((theta2 - theta3) ** 2.0 + 2.0 * (theta2 + theta3) ** 2.0 + 1.0)
    )
    # compute q
    q = domega_gamma1 / (-thetas)
    # stability conditions
    # - compute the quartic condition
    quartic = (coeffq0 + (coeffq2 * (q**2.0)) + (coeffq4 * (q**4.0))) > 0.0
    # - compute the general condition
    general = thetas > 0.0
    # return the overall stability (fulfilling both conditions is necessary)
    return quartic * general


# corrected dziembowski conditions with theta input
def corrected_dziembowski(
    theta2: 'npt.NDArray[np.float64]',
    theta3: 'npt.NDArray[np.float64]',
    domega_gamma1: 'float',
) -> 'npt.NDArray[np.bool_]':
    """Computes whether a stationary point of the AEs is stable, if it is hyperbolic.

    Notes
    -----
    Uses the stability conditions defined in Van Beeck et al. (2024).

    Parameters
    ----------
    theta2 : np.ndarray[np.float64]
        X-Part of the meshgrid containing values of the dimensionless theta_2 parameter (-gamma_2 / gamma_1).
    theta3 : np.ndarray[np.float64]
        Y-Part of the meshgrid containing values of the dimensionless theta_3 parameter (-gamma_3 / gamma_1).
    domega_gamma1 : float
        Fixed value for delta omega / gamma_1 (i.e. linear frequency detuning / linear parent driving rate).

    Returns
    -------
    np.ndarray[np.bool_]
        True if stable, False if not.
    """
    # compute intermediate values
    theta23 = theta2 * theta3
    thetas = -theta2 - theta3 + 1.0
    # compute the coefficients for the quartic condition
    coeffq0 = (thetas) ** 3.0 - 2.0 * theta23
    coeffq2 = -12.0 * theta23 - thetas * (
        (theta2 - theta3) ** 2.0 + (theta2 + theta3) ** 2.0 + 2.0
    )
    coeffq4 = -18.0 * theta23 - (3.0 * thetas) * (
        (thetas**2.0) + 4.0 * (theta2 + theta3 - theta23)
    )
    # compute q
    q = domega_gamma1 / thetas
    # stability conditions
    # - compute the quartic condition
    quartic = (coeffq0 + (coeffq2 * (q**2.0)) + (coeffq4 * (q**4.0))) > 0.0
    # - compute the general condition
    general = thetas < 0.0
    # return the overall stability (fulfilling both conditions is necessary)
    return quartic * general


# adjust ticks
def adjust_ticks(ax: 'Axes', xtickbottom: 'bool', ytickleft: 'bool') -> 'None':
    """Adjust ticks for the input Axes object.

    Parameters
    ----------
    ax : Axes
        The Axes object for which the ticks will be adjusted.
    xtickbottom : bool
        If True, put x-ticks on the bottom x-axis. If False, put them on top x-axis.
    ytickleft : bool
        If True, put y-ticks on the left y-axis. If False, put them on the right y-axis.
    """
    # locators
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # set tick labels to different locations, if preferred
    if not xtickbottom:
        ax.xaxis.tick_top()
    if not ytickleft:
        ax.yaxis.tick_right()
    # ticks
    ax.tick_params(
        top=True, bottom=True, right=True, left=True, which='both', labelsize=8
    )


def adjust_labels(
    plot_xlabel: 'bool', plot_ylabel: 'bool', ax: 'Axes'
) -> 'None':
    """Adjust the plot labels.

    Parameters
    ----------
    plot_xlabel : bool
        Set to True if you want the x-label to be plotted.
    plot_ylabel : bool
        Set to True if you want the y-label to be plotted.
    ax : Axes
        The Axes object for which the labels will be adjusted.
    """
    # labels
    if plot_xlabel:
        ax.set_xlabel(r'$\vartheta_2$', size=10)
    if plot_ylabel:
        ax.set_ylabel(r'$\vartheta_3$', size=10)
    # rotate x-tick labels, if necessary
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center')


def add_text_label(
    ax: 'Axes',
    delta_omega_over_gamma_1: 'float',
    text_color: 'str',
) -> 'None':
    """Adds a text label representing the value of the

    Parameters
    ----------
    ax : Axes
        Axes object to which the text label shall be added.
    delta_omega_over_gamma_1 : float
        Ratio of the linear frequency detuning to the linear driving rate of the parent mode.
    text_color : str
        Color of the to-be-added text label.
    """
    # generate the exponent value for the text label
    _mod_exp = np.log10(delta_omega_over_gamma_1)
    # add the label to the axes object
    ax.text(
        0.20,
        0.50,
        rf'$\left(\dfrac{{\delta\omega}}{{\delta\gamma_1}}\right) = 10^{{{_mod_exp:.0f}}}$',
        horizontalalignment='left',
        verticalalignment='center',
        color=text_color,
        alpha=0.8,
        transform=ax.transAxes,
    )


def get_custom_colormap() -> 'ListedColormap':
    """Sets up the custom colormap for the generated figures in Van Beeck et al. (2024).

    Returns
    -------
    ListedColormap
        Customized color map for the generated figures.
    """
    return ListedColormap(['white', 'k'])


def get_dict_axes_information(ax: 'NDArrayOfAxes') -> 'AxesInformationDict':
    """Returns a dictionary of data related to the setup of the axes, as well as the plot style of the generated figure,
    based on an input array of axes objects.

    Parameters
    ----------
    ax : np.ndarray[Axes]
        Array containing axes objects for the generated figure.

    Returns
    -------
    AxesInformationDict
        Contains information on the axes objects that is necessary to generate the figure.
    """
    # return the dictionary containing information regarding the setup of the axes and the plot style
    return {
        'ax_list': [ax[0], ax[1], ax[2]],
        'plot_xlabel': [True, True, True],
        'plot_ylabel': [True, False, False],
        'y_left': [True, True, True],
        'x_bottom': [True, True, True],
        'text_colors': ['k', 'white', 'white'],
    }


def get_figure_information() -> 'tuple[AxesInformationDict, Figure]':
    """Returns a dictionary containing data related to axes setup and plot style (of the generated figure),
    and the figure object itself.

    Returns
    -------
    axes_info_dict : AxesInformationDict
        Contains information on the axes objects that is necessary to generate the figure.
    fig : Figure
        Figure object.
    """
    # initialize the figure and axes objects
    fig, ax = plt.subplots(1, 3, figsize=(6.0, 9.0), dpi=300)
    # get the axes information dictionary
    axes_info_dict = get_dict_axes_information(ax=ax)
    # return the axes information dictionary and the figure object
    return axes_info_dict, fig


def get_grid_information(first_row: 'bool') -> 'GridInformationDict':
    """Returns a dictionary containing data related to the meshgrid setup for the calculations.

    Parameters
    ----------
    first_row : bool
        Specifies whether to generate the first or second row of panels
        of figure 1 of Van Beeck et al. (2024).

    Returns
    -------
    GridInformationDict
        Contains information on the parameters necessary to set up the computational grid.
    """
    if first_row:
        return {
            'domega_gamma1': [1e-1, 1e0, 1e1],
            'theta_min': [1.0e-2, 1.0e-1, 1.0e-1],
            'theta_max': [3.0, 10.0, 40.0],
            'dtheta': [0.0025, 0.0025, 0.01],
        }
    else:
        return {
            'domega_gamma1': [1e2, 1e3, 1e4],
            'theta_min': [0.1, 0.1, 0.1],
            'theta_max': [280.0, 750.0, 8000.0],
            'dtheta': [0.05, 0.5, 5.0],
        }


def generate_loop_data(
    grid_info: 'GridInformationDict', axes_info: 'AxesInformationDict'
) -> 'LoopDataDict':
    """Generates the data dictionary used to fill the subplots with information.

    Parameters
    ----------
    grid_info : GridInformationDict
        Contains information on the parameters necessary to set up the computational grid.
    axes_info : AxesInformationDict
        Contains information on the axes objects that is necessary to generate the figure.

    Returns
    -------
    LoopDataDict
        Contains the necessary information used to fill the subplots of the figure.
    """
    return grid_info | axes_info  # type: ignore    return grid_info | axes_info


def create_figure_meta_data(grid_data: 'GridInformationDict') -> 'MetaDataDict':
    """Generates the figure metadata.

    Parameters
    ----------
    grid_data : GridInformationDict
        Dictionary containing the necessary information for the metadata,
        the ratios of linear frequency detunings to linear driving rate
        of the parent mode.

    Returns
    -------
    MetaDataDict
        Contains the figure metadata.
    """
    # create metadata for the figure
    return {
        'Title': 'Stability domains of AE fixed points for various values of dg1.',
        'Author': 'Jordan Van Beeck',
        'Description': f'Describes the stability domains of the fixed points '
        f'of the amplitude equations for a specific value of '
        f'the ratio of the linear frequency detuning to the '
        f'linear driving rate of the parent mode '
        f'(i.e. delta omega / gamma_1). That ratio is for this '
        f'figure equal to {[f"{x:.2E}" for x in grid_data["domega_gamma1"]]}.',
        'Software': f'Made with Python version '
        f'{sys.version_info.major}.{sys.version_info.minor}, '
        f'making use of the packages Matplotlib '
        f'(version {matplotlib.__version__}) '
        f'and Numpy (version {np.version.version}).',
    }


def generate_save_name(add_text: 'bool', first_row: 'bool') -> 'str':
    """Generates the name of the saved figure.

    Parameters
    ----------
    add_text : bool
        Determines whether to add explanatory text in the figure.
    first_row : bool
        Specifies whether to generate the first or second row of panels
        of figure 1 of Van Beeck et al. (2024).

    Returns
    -------
    str
        Name of the figure to be saved.
    """
    match (add_text, first_row):
        case (True, False):
            return 'Stability_domain_dg1_subplot_e2_e3_e4_paper_txt.png'
        case (True, True):
            return 'Stability_domain_dg1_subplot_e-1_e0_e1_paper_txt.png'
        case (False, False):
            return 'Stability_domain_dg1_subplot_e2_e3_e4_paper.png'
        case _:
            return 'Stability_domain_dg1_subplot_e-1_e0_e1_paper.png'


def save_figure(
    add_text: 'bool', meta: 'MetaDataDict', first_row: 'bool'
) -> 'None':
    """Saves the generated figure.

    Parameters
    ----------
    add_text : bool
        Determines whether to add explanatory text in the figure.
    meta : MetaDataDict
        Contains the figure metadata.
    first_row : bool
        Specifies whether to generate the first or second row of panels
        of figure 1 of Van Beeck et al. (2024).
    """
    # create the save name
    save_name = generate_save_name(add_text=add_text, first_row=first_row)
    # save the figure
    plt.savefig(
        save_name,
        dpi='figure',
        format='png',
        metadata=meta,
        bbox_inches='tight',
        pad_inches=0.01,
        transparent=True,
    )


if __name__ == '__main__':
    # ----------------------
    # USER INPUT INFORMATION
    # ----------------------
    # specify whether to add explanatory text to the figures
    add_text = False
    # specify whether to generate the first or second row of panels
    # of figure 1 of Van Beeck et al. (2024)
    # --> stitching both the figures containing the rows of panels
    # was done using LaTeX
    first_row = True
    # ------------------------------------------------
    # FIXED USER INPUT INFORMATION FOR REPRODUCIBILITY
    # ------------------------------------------------
    # generate grid-specific information
    grid_data = get_grid_information(first_row=first_row)
    # generate figure-specific information and a figure object
    figure_data, fig = get_figure_information()
    # get the custom colormap
    custom_cmp = get_custom_colormap()
    # -------------------------------
    # FILL THE SUBPLOTS OF THE FIGURE
    # -------------------------------
    # generate loop data
    loop_data = generate_loop_data(grid_info=grid_data, axes_info=figure_data)
    # fill subplots
    for dg1, tmi, tma, dth, _ax, p_xl, p_yl, yle, xbo, _col in zip(
        *loop_data.values()
    ):
        # vary the values of theta_2 and theta_3 in a grid
        X, Y, extent = comp_meshgrid(theta_min=tmi, theta_max=tma, dtheta=dth)
        # compute whether the stationary points are stable (if they are hyperbolic),
        # according to Dziembowski and the updated Van Beeck et al. (2024) criteria
        Z_Dziembowski = original_dziembowski(X, Y, dg1)
        Z_Van_Beeck = corrected_dziembowski(X, Y, dg1)
        # SHOW THE DZIEMBOWSKI (1982) CONTOURS OF STABILITY
        im = _ax.contour(X, Y, Z_Dziembowski, 1, linewidths=0.1, colors='k')
        matplotlib.rc('hatch', color='k', linewidth=0.1)
        im = _ax.contourf(
            X, Y, Z_Dziembowski, 1, colors='None', hatches=[None, '/']
        )
        # SHOW THE UPDATED VAN BEECK ET AL. (2024) CONTOURS OF STABILITY
        im = _ax.imshow(
            Z_Van_Beeck,
            interpolation='bilinear',
            cmap=custom_cmp,
            origin='lower',
            extent=extent,
            vmax=1,
            vmin=0,
            alpha=0.85,
        )
        # adjust ticks
        adjust_ticks(ax=_ax, xtickbottom=xbo, ytickleft=yle)
        # adjust labels
        adjust_labels(plot_xlabel=p_xl, plot_ylabel=p_yl, ax=_ax)
        # if requested, add a text label
        if add_text:
            add_text_label(
                ax=_ax, delta_omega_over_gamma_1=dg1, text_color=_col
            )
        # delete arrays to make space
        del X
        del Y
        del Z_Dziembowski
        del Z_Van_Beeck
    # set tight layout on the figure
    fig.tight_layout()
    # create metadata for the figure
    meta = create_figure_meta_data(grid_data=grid_data)
    # save figure
    save_figure(add_text=add_text, meta=meta, first_row=first_row)
    # show
    plt.show()
