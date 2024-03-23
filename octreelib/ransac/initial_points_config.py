"""
This file contains the configuration for the number of initial points to be used in the RANSAC algorithm.
This number is used to generate random indices for the initial points.
It can be configured to generate different number of initial points.
It is used by functions inside both the RANSAC class and the util module
and because of that it is placed in a separate file.
"""

N_INITIAL_POINTS = 6
