'''Python script used to compute/interpolate the threshold boundary for theta_2, based on values of the isolated mode triads listed in Table 5 of Van Beeck et al. (forthcoming).

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt


# store the theta_2 values
theta_2 = [0.76947, 0.92560, 0.30289, 0.17994, 2.06433, 1.27865, 0.01816, 0.76510, 0.13080, 0.00907, 0.02771, 0.85748, 0.09346, 38.7770, 3.58916, 0.21492, 0.92741, 0.99670, 0.41545, 1.58532, 18.0122]
# store the daughter-parent amplitude ratio (daughter 1)
ratio_2_1 = [0.55605, 0.42858, 0.65585, 0.91831, 0.48298, 0.32000, 2.53183, 0.32190, 0.84057, 3.08804, 2.02044, 0.79629, 2.05217, 0.07627, 0.29260, 0.72069, 0.34623, 0.48095, 0.61223, 0.30289, 0.12459]


# store the data in a dataframe
my_df = pd.DataFrame(data=zip(theta_2, ratio_2_1), columns=['theta_2', 'ratio_2_1'])


# get the necessary difference arrays
my_diff = my_df.loc[:, 'ratio_2_1'].to_numpy() - 1.0
pos = my_diff > 0.0
my_pos_diff = my_diff.copy()
my_neg_diff = my_diff.copy()
my_pos_diff[~pos] = np.NaN
my_neg_diff[pos] = np.NaN


# get the k smallest point masks on both sides
k = 2
pos_smallest_k = np.argpartition(my_pos_diff, k)
neg_smallest_k = np.argpartition(np.abs(my_neg_diff), k)


# plot the data points
my_df.plot(x='theta_2', y='ratio_2_1', kind='scatter')


# store data points in arrays
X = np.array(theta_2)
Y = np.array(ratio_2_1)


# get 3 closest points on both sides
xx_k = np.concatenate([X[pos_smallest_k[:k]], X[neg_smallest_k[:k]]])
yy_k = np.concatenate([Y[pos_smallest_k[:k]], Y[neg_smallest_k[:k]]])


# perform linear regression
lin_result = linregress(yy_k, xx_k)


# compute the linear regression result for the 1.0 boundary
uncertainty_boundary = np.sqrt(lin_result.stderr**2.0 + lin_result.intercept_stderr**2.0)
boundary = lin_result.intercept + lin_result.slope


# print the result
print(f'Linear regression for boundary yields: {boundary} +/- {uncertainty_boundary}')
print(f'regression slope: {lin_result.slope} +/- {lin_result.stderr}')
print(f'regression intercept: {lin_result.intercept} +/- {lin_result.intercept_stderr}')


# plot the regression result
plt.plot(lin_result.intercept + lin_result.slope*yy_k, yy_k, 'r')


# show the plots
plt.show()
