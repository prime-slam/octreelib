"""
These functions are auxiliary functions for the RANSAC algorithm.
They cannot be defined inside the `CudaRansac` class because
`CudaRansac.__ransac_kernel` would not be able to access them.
This file also contains the `INITIAL_POINTS_NUMBER` constant which
can be used to configure the number of initial points to be used in the RANSAC algorithm.
"""

import math
import numba as nb

from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

# This constant configures the number of initial points to be used in the RANSAC algorithm.
INITIAL_POINTS_NUMBER = 6


@cuda.jit(
    device=True,
    inline=True,
)
def measure_distance(plane, point):
    """
    Measure the distance between a plane and a point.
    :param plane: Plane coefficients.
    :param point: Point coordinates.
    """
    return (
        math.fabs(
            plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]
        )
        / (plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2) ** 0.5
    )


@cuda.jit(device=True, inline=True)
def generate_random_int(rng_states, lower_bound, upper_bound):
    """
    Generate a random number between a and b.
    :param rng_states: Random number generator states.
    :param lower_bound: Lower bound.
    :param upper_bound: Upper bound.
    """
    thread_id = cuda.grid(1)
    x = xoroshiro128p_uniform_float32(rng_states, thread_id)
    return nb.int32(x * (upper_bound - lower_bound) + lower_bound)


@cuda.jit(device=True, inline=True)
def generate_random_indices(
    initial_point_indices, rng_states, block_size, points_number
):
    """
    Generate random points from the given block.
    :param initial_point_indices: Array to store the initial point indices.
    :param rng_states: Random number generator states.
    :param block_size: Size of the block.
    :param points_number: Number of points to generate.
    """

    for i in range(points_number):
        initial_point_indices[i] = generate_random_int(rng_states, 0, block_size)
    return initial_point_indices


@cuda.jit(device=True, inline=True)
def get_plane_from_points(points, initial_point_indices):
    """
    Calculate the plane coefficients from the given points.
    :param points: Point cloud.
    :param initial_point_indices: Inliers to calculate the plane coefficients from.
    """

    centroid_x, centroid_y, centroid_z = 0.0, 0.0, 0.0

    for idx in initial_point_indices:
        centroid_x += points[idx][0]
        centroid_y += points[idx][1]
        centroid_z += points[idx][2]

    centroid_x /= INITIAL_POINTS_NUMBER
    centroid_y /= INITIAL_POINTS_NUMBER
    centroid_z /= INITIAL_POINTS_NUMBER

    xx, xy, xz, yy, yz, zz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for idx in initial_point_indices:
        r_x = points[idx][0] - centroid_x
        r_y = points[idx][1] - centroid_y
        r_z = points[idx][2] - centroid_z
        xx += r_x * r_x
        xy += r_x * r_y
        xz += r_x * r_z
        yy += r_y * r_y
        yz += r_y * r_z
        zz += r_z * r_z

    det_x = yy * zz - yz * yz
    det_y = xx * zz - xz * xz
    det_z = xx * yy - xy * xy

    if det_x > det_y and det_x > det_z:
        abc_x = det_x
        abc_y = xz * yz - xy * zz
        abc_z = xy * yz - xz * yy
    elif det_y > det_z:
        abc_x = xz * yz - xy * zz
        abc_y = det_y
        abc_z = xy * xz - yz * xx
    else:
        abc_x = xy * yz - xz * yy
        abc_y = xy * xz - yz * xx
        abc_z = det_z

    norm = (abc_x**2 + abc_y**2 + abc_z**2) ** 0.5
    if norm == 0:
        return 0.0, 0.0, 0.0, 0.0

    abc_x /= norm
    abc_y /= norm
    abc_z /= norm
    d = -(abc_x * centroid_x + abc_y * centroid_y + abc_z * centroid_z)
    return abc_x, abc_y, abc_z, d
