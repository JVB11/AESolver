"""Python script used to create stability-domain-visualizing figures for Van Beeck et al. (forthcoming).

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.colors import ListedColormap
import sys


# Use local TeX installation
plt.rcParams.update(
    {
        'text.usetex': True,
        'font.family': 'serif',
        'font.sans-serif': 'Times',
    }
)


# compute meshgrid and grid extent parameters
def comp_meshgrid(theta_min=1.0e-1, theta_max=100.0, dtheta=0.1):
    theta_range = np.arange(theta_min, theta_max, dtheta)
    X, Y = np.meshgrid(theta_range, theta_range)
    extent = theta_min, theta_max, theta_min, theta_max
    return X, Y, extent


# original dziembowski conditions with theta input
def original_dziembowski(theta2, theta3, domega_gamma1):
    """Computes whether a stationary point of the AEs is stable, if it is hyperbolic, as claimed by Dziembowski (1982).

    Notes
    -----
    Uses the stability conditions defined in Dziembowski (1982).

    Parameters
    ----------
    theta2 : np.ndarray
        X-Part of the meshgrid containing values of the dimensionless theta_2 parameter (-gamma_2 / gamma_1).
    theta3 : np.ndarray
        Y-Part of the meshgrid containing values of the dimensionless theta_3 parameter (-gamma_3 / gamma_1).
    domega_gamma1 : float
        Fixed value for delta omega / gamma_1 (i.e. linear frequency detuning / linear parent driving rate).

    Returns
    -------
    np.ndarray[bool]
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
def corrected_dziembowski(theta2, theta3, domega_gamma1):
    """Computes whether a stationary point of the AEs is stable, if it is hyperbolic.

    Notes
    -----
    Uses the stability conditions defined in Van Beeck et al. (2022).

    Parameters
    ----------
    theta2 : np.ndarray
        X-Part of the meshgrid containing values of the dimensionless theta_2 parameter (-gamma_2 / gamma_1).
    theta3 : np.ndarray
        Y-Part of the meshgrid containing values of the dimensionless theta_3 parameter (-gamma_3 / gamma_1).
    domega_gamma1 : float
        Fixed value for delta omega / gamma_1 (i.e. linear frequency detuning / linear parent driving rate).

    Returns
    -------
    np.ndarray[bool]
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
def adjust_ticks(ax, xtickbottom, ytickleft):
    """Adjust ticks for the input matplotlib.axes.Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object for which the ticks will be adjusted.
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


if __name__ == '__main__':
    # consider specific value of $\delta\omega / \gamma_1$
    # domega_gamma1 = [2e-2, 1e0, 1e1]
    # domega_gamma1 = [1e-1, 1e0, 1e1]
    # domega_gamma1 = [1e0, 1e1, 1e2]#[1e-1, 1e0, 1e1, 1e2]
    domega_gamma1 = [1e2, 1e3, 1e4]
    add_text = False
    # theta_min = [1.0e-3, 1.0e-1, 1.0e-1]
    # theta_min = [1.0e-2, 1.0e-1, 1.0e-1]
    # theta_min = [1.0e-1, 1.0e-1, 0.1]#[1.0e-2, 1.0e-1, 1.0e-1, 0.1]
    theta_min = [0.1, 0.1, 0.1]
    # theta_max = [1.5, 10.0, 40.0]
    # theta_max = [3.0, 10.0, 40.0]
    # theta_max = [10.0, 40.0, 280.0]#[3.0, 10.0, 40.0, 280.0]
    theta_max = [280.0, 750.0, 8000.0]
    # dtheta = [0.0001, 0.0025, 0.01]
    # dtheta = [0.0025, 0.0025, 0.01]
    # dtheta = [0.0025, 0.005, 0.05]#[0.005, 0.005, 0.01, 0.1]
    dtheta = [0.05, 0.5, 5.0]  # [0.005, 0.005, 0.01, 0.1]
    # custom colormap
    cmp = ListedColormap(['white', 'k'])
    # figure + axes
    fig, ax = plt.subplots(1, 3, figsize=(6.0, 9.0), dpi=300)
    ax_list = [ax[0], ax[1], ax[2]]  # [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]
    plot_xlabel = [True, True, True]  # [False, False, True, True]
    plot_ylabel = [True, False, False]  # [True, False, True, False]
    y_left = [True, True, True]  # [True, False, True, False]
    x_bot = [True, True, True]  # [False, False, True, True]
    text_colors = ['k', 'white', 'white']
    # loop to fill subplots
    for dg1, tmi, tma, dth, _ax, p_xl, p_yl, yle, xbo, _col in zip(
        domega_gamma1,
        theta_min,
        theta_max,
        dtheta,
        ax_list,
        plot_xlabel,
        plot_ylabel,
        y_left,
        x_bot,
        text_colors,
    ):
        # vary the values of theta_2 and theta_3 in a grid
        X, Y, extent = comp_meshgrid(theta_min=tmi, theta_max=tma, dtheta=dth)
        # compute whether the stationary point is stable (if hyperbolic)
        Z1 = original_dziembowski(X, Y, dg1)
        Z2 = corrected_dziembowski(X, Y, dg1)
        # original
        im = _ax.contour(X, Y, Z1, 1, linewidths=0.1, colors='k')
        matplotlib.rc('hatch', color='k', linewidth=0.1)
        im = _ax.contourf(X, Y, Z1, 1, colors='None', hatches=[None, '/'])
        # corrected
        im = _ax.imshow(
            Z2,
            interpolation='bilinear',
            cmap=cmp,
            origin='lower',
            extent=extent,
            vmax=1,
            vmin=0,
            alpha=0.85,
        )
        # adjust ticks
        adjust_ticks(_ax, xbo, yle)
        # labels
        if p_xl:
            _ax.set_xlabel(r'$\vartheta_2$', size=10)
        if p_yl:
            _ax.set_ylabel(r'$\vartheta_3$', size=10)
        # rotate x-tick labels, if necessary
        plt.setp(_ax.get_xticklabels(), rotation=90, ha='center')
        # add text label on plot, if wanted
        if add_text:
            _mod_exp = np.log10(dg1)
            _ax.text(
                0.20,
                0.50,
                rf'$\left(\dfrac{{\delta\omega}}{{\delta\gamma}}\right) = 10^{{{_mod_exp:.0f}}}$',
                horizontalalignment='left',
                verticalalignment='center',
                color=_col,
                alpha=0.8,
                transform=_ax.transAxes,
            )
        # delete arrays to make space
        del X
        del Y
        # del Z1
        del Z2
    # tight layout
    fig.tight_layout()
    # create metadata for the figure
    meta = {
        'Title': 'Stability domains of AE fixed points for various values of dg1.',
        'Author': 'Jordan Van Beeck',
        'Description': f'Describes the stability domains of the fixed points '
        f'of the amplitude equations for a specific value of '
        f'the ratio of the linear frequency detuning to the '
        f'linear driving rate of the parent mode '
        f'(i.e. delta omega / gamma_1). That ratio is for this '
        f'figure equal to {[f"{x:.2E}" for x in domega_gamma1]}.',
        'Software': f'Made with Python version '
        f'{sys.version_info.major}.{sys.version_info.minor}, '
        f'making use of the packages Matplotlib '
        f'(version {matplotlib.__version__}) '
        f'and Numpy (version {np.version.version}).',
    }
    # save figure
    save_name = (
        'Stability_domain_dg1_subplot_e2_e3_e4_paper_txt.png'
        if add_text
        else 'Stability_domain_dg1_subplot_e2_e3_e4_paper.png'
    )
    plt.savefig(
        save_name,
        dpi='figure',
        format='png',
        metadata=meta,
        bbox_inches='tight',
        pad_inches=0.01,
        transparent=True,
    )
    # show
    plt.show()
