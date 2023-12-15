'''Python module defining a mock mode info object to be used during testing of the aesolver.pre_checks module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# define the mockup class containing the mode information
class MockModeInfo:
    """Class that is a mockup of a mode info object.
    
    Parameters
    ----------
    l : int
        The spherical degree.
    m : int
        The azimuthal order.
    """
    # attribute type declarations
    l: int
    m: int
    
    def __init__(self, l: int, m: int) -> None:
        # set the spherical degree l
        self.l = l
        # set the azimuthal order m
        self.m = m
        
    # make it subscriptable
    def __getitem__(self, item: str) -> int:
        return getattr(self, item)
