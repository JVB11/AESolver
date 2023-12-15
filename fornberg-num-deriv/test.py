# import the module
from myfnd import fnd  # type: ignore

# import numpy to generate test arrays
import numpy as np


# test derivation of sin
def test_sin():
    # get the grid variable and the u variable (that needs to be differentiated)
    grid = np.linspace(0.0, 2.0 * np.pi, num=5000, dtype=np.float64)
    u = np.sin(grid)
    # get the expected derivative
    du_expected = np.cos(grid)
    # get the accuracy and order attributes
    acc = 10  # < 20
    order = 1  # derivative order (cannot be too high)
    # run the fortran code to attempt to numerically reconstruct the derivative
    du_computed = np.zeros_like(u)
    du_computed = np.asfortranarray(du_computed)
    fnd.fornberg_grid_derivative(
        order, acc, np.asfortranarray(grid), np.asfortranarray(u), du_computed
    )
    # print the results
    print(
        'The resulting reconstructed first derivative of the sine function is:'
    )
    print(du_computed)
    print('And the expected result is the cosine function:')
    print(du_expected)


if __name__ == '__main__':
    print('Now testing MYFND.')
    test_sin()
    print('Tests completed.')
