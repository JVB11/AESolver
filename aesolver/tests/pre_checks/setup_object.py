'''Python module defining a setup object used while testing the aesolver.pre_checks module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
class SetupObject:
    # attribute type declarations
    triads: list | None
    conjugation: list | None
    all_fulfilled: bool
        
    def __init__(self, triads: list | None,
                 conjugation: list | None,
                 all_fulfilled: bool) -> None:
        self.triads = triads
        self.conjugation = conjugation
        self.all_fulfilled = all_fulfilled
