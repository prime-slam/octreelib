"""
These functions are auxiliary functions for the RANSAC algorithm.
They cannot be defined inside the `CudaRansac` class because
`CudaRansac.__ransac_kernel` would not be able to access them.
"""

import math

from numba import cuda


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
    return math.fabs(
        plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]
    )


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

    centroid_x /= initial_point_indices.shape[0]
    centroid_y /= initial_point_indices.shape[0]
    centroid_z /= initial_point_indices.shape[0]

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
