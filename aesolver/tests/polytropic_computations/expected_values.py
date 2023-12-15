'''Python module used to provide expected values for the polytropic structure information computed with aesolver.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import numpy as np
# import pytest helper class
from pytest_util_classes import EVs


# define the expected values
class ExpectedValuesPolytropicComputations(EVs):
    """Stores the expected values for polytropic computations."""
    
    DRHO = np.array(
        [-0.01119058, -0.00716197, -0.00402861,
         -0.00179049, -0.00044762]
        ).tobytes()
    DRHO2 = np.array(
        [ 0.07043073, -0.00230206,  0.00104266, 
         0.00138999,  0.00061268]
        ).tobytes()
    DRHO3 = np.array(
        [-1.26948060e+00,  6.75545742e-03, 
         8.15000481e-04, -1.13858746e-03,
         -7.99846844e-04]
        ).tobytes()
    DP = np.array(
        [-2.98757204e-10, -1.52963688e-10, -6.45315561e-11,
         -1.91204611e-11, -2.39005763e-12]
        ).tobytes()
    DP2 = np.array(
        [ 1.91017887e-09, -3.00464388e-11,  2.74569361e-11, 
         1.96236311e-11, 4.46642020e-12]
        ).tobytes()
    DP3 = np.array(
        [-3.44496932e-08,  1.67499141e-10,  8.28918657e-12,
         -2.09014181e-11, -7.98279249e-12]
        ).tobytes()
    DGRAVPOT = np.array(
        [8.00916e-09, 8.00916e-09, 8.00916e-09, 8.00916e-09,
         8.00916e-09]
        ).tobytes()
    DGRAVPOT2 = np.array(
        [-4.88058187e-08,  4.57666286e-09,  5.96836442e-10,
         -2.21305737e-09, -2.95337775e-09]
        ).tobytes()
    DGRAVPOT3 = np.array(
        [ 8.79130453e-07, -4.37234755e-09, -1.46839537e-09, 
         7.72351475e-10, 1.44665452e-09]
        ).tobytes()
    P = np.array(
        [7.46893010e-10, 3.05927377e-10, 9.67973341e-11,
         1.91204611e-11, 1.19502882e-12]
        ).tobytes()
    RHO = np.array(
        [0.03730194, 0.01909859, 0.00805722, 0.00238732,
         0.00029842]
        ).tobytes()
    R = np.array(
        [0.2, 1.4, 2.6, 3.8, 5. ]
        ).tobytes()
    V2 = np.array(
        [50.        ,  8.92857143,  6.41025641, 
         6.57894737, 10.        ]
        ).tobytes()
    V = np.array(
        [ 0.08      ,  0.7       ,  1.73333333, 
         3.8       , 10.        ]
        ).tobytes()
    U = np.array(
        [0.78125, 2.8    , 2.19375, 0.95   , 0.15625]
        ).tobytes()
    C1 = np.array([0.04, 0.28, 0.52, 0.76, 1.  ]).tobytes()
    AS = np.array(
        [0.012, 0.105, 0.26 , 0.57 , 1.5  ]
        ).tobytes()
    MR = np.array(
        [0.0048, 0.2352, 0.8112, 1.7328, 3.    ]
        ).tobytes()
    SURFGRAV = np.array(
        [8.00916e-09, 8.00916e-09, 8.00916e-09, 8.00916e-09,
         8.00916e-09]
        ).tobytes()
    BRUNT_N2 = np.array(
        [4.805496e-10, 6.006870e-10, 8.009160e-10,
         1.201374e-09, 2.402748e-09]
        ).tobytes()
    P0 = 3.0592737696333214e-10
    RHO0 = 0.01909859317102744
    