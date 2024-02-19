import math

import numpy as np
import numpy.typing as npt
import numba as nb
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_uniform_float32

from octreelib.internal import PointCloud


@cuda.jit(device=True, inline=True)
def _cu_subtract_point(ax, ay, az, bx, by, bz):
    return ax - bx, ay - by, az - bz


@cuda.jit(
    "float32(float32, float32, float32, float32, float32, float32)",
    device=True,
    inline=True,
)
def _cu_dot(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


@cuda.jit(device=True, inline=True)
def _crossNormal(ax, ay, az, bx, by, bz):
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    s = math.sqrt(cx * cx + cy * cy + cz * cz)
    return cx / s, cy / s, cz / s


@cuda.jit(device=True, inline=True)
def _cuRand(rng_states, a, b):
    thread_id = cuda.grid(1)
    x = xoroshiro128p_uniform_float32(rng_states, thread_id)
    return (nb.int32)(x * (b - a) + a)


@cuda.jit(device=True)
def _process_inliers(inlier_count, this_model, best_model):
    if inlier_count > best_model[0]:
        best_model[0] = inlier_count
        best_model[1] = this_model[0]
        best_model[2] = this_model[1]
        best_model[3] = this_model[2]
        best_model[4] = this_model[3]


@cuda.jit(device=True, inline=True)
def get_plane_from_points(points, inliers):
    centroid_x = 0.0
    centroid_y = 0.0
    centroid_z = 0.0

    for idx in inliers:
        for i in range(3):
            centroid_x += points[idx][0]
            centroid_y += points[idx][1]
            centroid_z += points[idx][2]

    num_inliers = len(inliers)
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


@cuda.jit(device=True, inline=True)
def get_plane_from_points_old(points, inliers):
    # centroid = np.zeros(3, np.float64)
    centroid = cuda.local.array(shape=3, dtype=nb.float32)
    centroid[0] = 0
    centroid[1] = 0
    centroid[2] = 0
    for idx in inliers:
        # centroid += points[idx]
        centroid[0] += points[idx][0]
        centroid[1] += points[idx][1]
        centroid[2] += points[idx][2]
    centroid[0] /= len(inliers)
    centroid[1] /= len(inliers)
    centroid[2] /= len(inliers)

    xx = 0.0
    xy = 0.0
    xz = 0.0
    yy = 0.0
    yz = 0.0
    zz = 0.0

    for idx in inliers:
        r = cuda.local.array(shape=3, dtype=nb.float32)
        r[0] = points[idx][0] - centroid[0]
        r[1] = points[idx][1] - centroid[1]
        r[2] = points[idx][2] - centroid[2]
        # r = points[idx] - centroid
        xx += r[0] * r[0]
        xy += r[0] * r[1]
        xz += r[0] * r[2]
        yy += r[1] * r[1]
        yz += r[1] * r[2]
        zz += r[2] * r[2]

    det_x = yy * zz - yz * yz
    det_y = xx * zz - xz * xz
    det_z = xx * yy - xy * xy

    if det_x > det_y and det_x > det_z:
        abc = np.array([det_x, xz * yz - xy * zz, xy * yz - xz * yy])
    elif det_y > det_z:
        abc = np.array([xz * yz - xy * zz, det_y, xy * xz - yz * xx])
    else:
        abc = np.array([xy * yz - xz * yy, xy * xz - yz * xx, det_z])

    norm = np.linalg.norm(abc)
    if norm == 0:
        # plane_coefficients[0] = 0
        # plane_coefficients[1] = 0
        # plane_coefficients[2] = 0
        # plane_coefficients[3] = 0
        return 0, 0, 0, 0
    else:
        abc /= norm
        d = -np.dot(abc, centroid)

        # plane_coefficients[0] = abc[0]
        # plane_coefficients[1] = abc[1]
        # plane_coefficients[2] = abc[2]
        # plane_coefficients[3] = d
        return abc[0], abc[1], abc[2], d


@cuda.jit
def _do_fit(
    points: PointCloud,
    block_sizes: npt.NDArray,
    block_start_indices: npt.NDArray,
    threshold: float,
    n_initial_points: int,
    result_mask: npt.NDArray,
    rng_states,
):
    # thread_id = cuda.grid(1)

    (i, j, k) = (cuda.threadIdx.x, cuda.blockIdx.x, cuda.blockDim.x)

    if block_sizes[j] < 3:
        result_mask[i][
            block_start_indices[j] : block_start_indices[j] + block_sizes[j]
        ] = 1
        return

    w = cuda.local.array(shape=4, dtype=nb.float32)
    if n_initial_points == 3:
        # !! this is the original implementation !!
        # !! original implementation only works with 3 initial points !!
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
        ux, uy, uz = _cu_subtract_point(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
        vx, vy, vz = _cu_subtract_point(p2[0], p2[1], p2[2], p3[0], p3[1], p3[2])
        w[0], w[1], w[2] = _crossNormal(ux, uy, uz, vx, vy, vz)
        w[3] = -_cu_dot(w[0], w[1], w[2], p1[0], p1[1], p1[2])
    else:
        # !! this is the new implementation !!
        # !! it is supposed to work with any number of initial points but does not work yet !!
        initial_points_indices = cuda.local.array(shape=6, dtype=nb.size_t)
        for ii in range(n_initial_points):
            initial_points_indices[ii] = (
                _cuRand(rng_states, 0, block_sizes[j]) + block_start_indices[j]
            )
        w[0], w[1], w[2], w[3] = get_plane_from_points(points, initial_points_indices)

    for jj in range(block_sizes[j]):
        p = points[block_start_indices[j] + jj]
        distance = math.fabs(_cu_dot(w[0], w[1], w[2], p[0], p[1], p[2]) + w[3]) / (
            w[0] ** 2 + w[1] ** 2 + w[2] ** 2
        )
        if distance < threshold:
            result_mask[i][block_start_indices[j] + jj] = 1
        else:
            result_mask[i][block_start_indices[j] + jj] = 0


class CudaRansac:
    def __init__(
        self,
        threshold: float = 0.01,
        initial_points: int = 6,
        iterations: int = 5000,
        n_blocks: int = 1,
        n_threads_per_block: int = 1,
        debug: bool = False,
    ) -> None:
        if threshold <= 0:
            raise ValueError("Threshold must be positive")
        if initial_points < 3:
            raise ValueError("Initial points count must be more or equal than three")
        if iterations < 1:
            raise ValueError("Number of RANSAC iterations must be positive")

        self.__threshold: float = threshold
        self.__initial_points: int = initial_points
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
        result_mask = np.zeros(
            (self.n_threads_per_block, len(point_cloud)), dtype=np.bool_
        )
        result_mask_cuda = cuda.to_device(result_mask)
        point_cloud_cuda = cuda.to_device(point_cloud)
        block_sizes_cuda = cuda.to_device(block_sizes)
        block_start_indices_cuda = cuda.to_device(block_start_indices)

        _do_fit[self.n_blocks, self.n_threads_per_block](
            point_cloud_cuda,
            block_sizes_cuda,
            block_start_indices_cuda,
            self.__threshold,
            self.__initial_points,
            result_mask_cuda,
            self.rng_states,
        )
        result_mask = result_mask_cuda.copy_to_host()
        print(result_mask)
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
        for block_start, block_size in zip(block_start_indices, block_sizes):
            print(
                np.argmax(
                    result_mask[:, block_start : block_start + block_size].sum(axis=1),
                    axis=0,
                ),
                np.max(
                    result_mask[:, block_start : block_start + block_size].sum(axis=1),
                    axis=0,
                )
                / block_size,
            )
        return maximum_mask
