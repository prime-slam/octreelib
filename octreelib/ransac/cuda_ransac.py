import math

import numpy as np
import numpy.typing as npt
import numba as nb
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_uniform_float32

from octreelib.internal import PointCloud


N_INITIAL_POINTS = 6


@cuda.jit(device=True, inline=True)
def _cu_subtract_point(ax, ay, az, bx, by, bz):
    """
    Subtract two points. (better be rewritten)
    :param ax: X component of the first point.
    :param ay: Y component of the first vector.
    :param az: Z component of the first vector.
    :param bx: X component of the second vector.
    :param by: Y component of the second vector.
    :param bz: Z component of the second vector.
    """
    return ax - bx, ay - by, az - bz


@cuda.jit(
    "float32(float32, float32, float32, float32, float32, float32)",
    device=True,
    inline=True,
)
def _cu_dot(ax, ay, az, bx, by, bz):
    """
    Calculate the dot product of two vectors. (better be rewritten)
    :param ax: X component of the first vector.
    :param ay: Y component of the first vector.
    :param az: Z component of the first vector.
    :param bx: X component of the second vector.
    :param by: Y component of the second vector.
    :param bz: Z component of the second vector.
    """
    return ax * bx + ay * by + az * bz


@cuda.jit(device=True, inline=True)
def _crossNormal(ax, ay, az, bx, by, bz):
    """
    Calculate the cross product of two vectors and normalize the result. (better be rewritten)
    :param ax: X component of the first vector.
    :param ay: Y component of the first vector.
    :param az: Z component of the first vector.
    :param bx: X component of the second vector.
    :param by: Y component of the second vector.
    :param bz: Z component of the second vector.
    """
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    s = math.sqrt(cx * cx + cy * cy + cz * cz)
    return cx / s, cy / s, cz / s


@cuda.jit(device=True, inline=True)
def _cuRand(rng_states, a, b):
    """
    Generate a random number between a and b.
    :param rng_states: Random number generator states.
    :param a: Lower bound.
    :param b: Upper bound.
    """
    thread_id = cuda.grid(1)
    x = xoroshiro128p_uniform_float32(rng_states, thread_id)
    return (nb.int32)(x * (b - a) + a)


@cuda.jit(device=True, inline=True)
def get_plane_from_points(points, inliers):
    """
    Calculate the plane coefficients from the given points.
    :param points: Point cloud.
    :param inliers: Inliers to calculate the plane coefficients from.
    """
    # This implementation works the same way as open3d implementation
    # ! but with some tweaks to make it work with numba and cuda !

    centroid_x = 0.0
    centroid_y = 0.0
    centroid_z = 0.0

    for idx in inliers:
        centroid_x += points[idx][0]
        centroid_y += points[idx][1]
        centroid_z += points[idx][2]

    num_inliers = N_INITIAL_POINTS
    centroid_x /= num_inliers
    centroid_y /= num_inliers
    centroid_z /= num_inliers

    xx = 0
    xy = 0
    xz = 0
    yy = 0
    yz = 0
    zz = 0

    for idx in inliers:
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


@cuda.jit
def _do_fit(
    points: PointCloud,
    block_sizes: npt.NDArray,
    block_start_indices: npt.NDArray,
    threshold: float,
    result_mask: npt.NDArray,
    rng_states,
):
    # thread_id = cuda.grid(1)

    (i, j, k) = (cuda.threadIdx.x, cuda.blockIdx.x, cuda.blockDim.x)

    if block_sizes[j] < N_INITIAL_POINTS:
        result_mask[i][
            block_start_indices[j] : block_start_indices[j] + block_sizes[j]
        ] = 0
        return

    w = cuda.local.array(shape=4, dtype=nb.float32)
    if N_INITIAL_POINTS == 3:
        # !! this is the original implementation !!
        # !! original implementation only works with 3 initial points !!

        # choose 3 random points
        i1 = _cuRand(rng_states, 0, block_sizes[j])
        i2 = _cuRand(rng_states, 0, block_sizes[j] - 1)
        if i2 >= i1:
            i2 = (i2 + 1) % block_sizes[j]
        i3 = _cuRand(rng_states, 0, block_sizes[j] - 2)
        if i3 >= i1:
            i3 = (i3 + 1) % block_sizes[j]
        if i3 >= i2:
            i3 = (i3 + 1) % block_sizes[j]
        (p1, p2, p3) = (
            points[block_start_indices[j] + i1],
            points[block_start_indices[j] + i2],
            points[block_start_indices[j] + i3],
        )

        # calculate the plane coefficients
        ux, uy, uz = _cu_subtract_point(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
        vx, vy, vz = _cu_subtract_point(p2[0], p2[1], p2[2], p3[0], p3[1], p3[2])
        w[0], w[1], w[2] = _crossNormal(ux, uy, uz, vx, vy, vz)
        w[3] = -_cu_dot(w[0], w[1], w[2], p1[0], p1[1], p1[2])
    else:
        # !! this is the new implementation !!
        # !! it is supposed to work with any number of initial points, but it works poorly !!

        # choose n_initial_points random points (does not work yet)
        initial_point_indices = cuda.local.array(
            shape=N_INITIAL_POINTS, dtype=nb.size_t
        )

        # for ii in range(N_INITIAL_POINTS):
        #     initial_point_indices[ii] = (
        #         _cuRand(rng_states, 0, block_sizes[j] - ii)
        #     )
        #     for _ in range(ii):
        #         for jj in range(ii):
        #             if initial_point_indices[ii] >= initial_point_indices[jj]:
        #                 initial_point_indices[ii] = (initial_point_indices[ii] + 1) % block_sizes[j]
        #     initial_point_indices_global[i + 1024 * j][ii] = initial_point_indices[ii]

        for ii in range(N_INITIAL_POINTS):
            initial_point_indices[ii] = _cuRand(rng_states, 0, block_sizes[j])
            unique = False
            while not unique:
                unique = True
                for jj in range(ii):
                    if initial_point_indices[ii] == initial_point_indices[jj]:
                        unique = False
                if not unique:
                    initial_point_indices[ii] = (
                        initial_point_indices[ii] + 1
                    ) % block_sizes[j]
        for ii in range(N_INITIAL_POINTS):
            initial_point_indices[ii] = (
                block_start_indices[j] + initial_point_indices[ii]
            )

        # calculate the plane coefficients
        w[0], w[1], w[2], w[3] = get_plane_from_points(points, initial_point_indices)

    # for each point in the block check if it is an inlier
    for jj in range(block_sizes[j]):
        p = points[block_start_indices[j] + jj]
        distance = (
            math.fabs(_cu_dot(w[0], w[1], w[2], p[0], p[1], p[2]) + w[3])
            / (w[0] ** 2 + w[1] ** 2 + w[2] ** 2) ** 0.5
        )
        if distance < threshold:
            result_mask[i][block_start_indices[j] + jj] = 1
        else:
            result_mask[i][block_start_indices[j] + jj] = 0


class CudaRansac:
    """
    RANSAC implementation using CUDA.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        iterations: int = 1024,
        n_blocks: int = 1,
        n_threads_per_block: int = 1,
        debug: bool = False,
    ) -> None:
        # in this implementation the parameters are set in the constructor
        # the alternative would be to set them in the fit method
        if threshold <= 0:
            raise ValueError("Threshold must be positive")
        if iterations < 1:
            raise ValueError("Number of RANSAC iterations must be positive")

        self.__threshold: float = threshold
        self.__iterations: int = iterations
        self.__debug: bool = debug
        self.rng_states = create_xoroshiro128p_states(
            n_threads_per_block * n_blocks, seed=0
        )
        self.n_blocks = n_blocks
        self.n_threads_per_block = n_threads_per_block

    def fit(
        self,
        point_cloud: PointCloud,
        block_sizes: npt.NDArray,
        block_start_indices: npt.NDArray,
    ):
        """
        Fit the model to the point cloud.
        :param point_cloud: Point cloud to fit the model to.
        :param block_sizes: Array of block sizes (should equal number of leaf voxels).
        :param block_start_indices: Array of block start indices (leaf voxel separators).
        """

        # create result mask and copy it to the device
        result_mask = np.zeros(
            (self.n_threads_per_block, len(point_cloud)), dtype=np.bool_
        )
        result_mask_cuda = cuda.to_device(result_mask)

        # copy point cloud, block sizes and block start indices to the device
        point_cloud_cuda = cuda.to_device(point_cloud)
        block_sizes_cuda = cuda.to_device(block_sizes)
        block_start_indices_cuda = cuda.to_device(block_start_indices)

        # call the kernel
        _do_fit[self.n_blocks, self.n_threads_per_block](
            point_cloud_cuda,
            block_sizes_cuda,
            block_start_indices_cuda,
            self.__threshold,
            result_mask_cuda,
            self.rng_states,
        )

        # copy result mask back to the host
        result_mask = result_mask_cuda.copy_to_host()

        # find the maximum mask individually for each leaf voxel and concatenate them
        maximum_mask = np.concatenate(
            [
                result_mask[:, block_start : block_start + block_size][
                    np.argmax(
                        result_mask[:, block_start : block_start + block_size].sum(
                            axis=1
                        ),
                        axis=0,
                    )
                ]
                for block_start, block_size in zip(block_start_indices, block_sizes)
            ]
        )
        return maximum_mask
