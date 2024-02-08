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


@cuda.jit
def _do_fit(
    points: PointCloud,
    block_sizes: npt.NDArray,
    block_start_indices: npt.NDArray,
    threshold: float,
    result_mask: npt.NDArray,
    rng_states,
):
    thread_id = cuda.grid(1)

    (i, j, k) = (cuda.threadIdx.x, cuda.blockIdx.x, cuda.blockDim.x)

    i1 = _cuRand(rng_states, 0, block_sizes[j])
    i2 = _cuRand(rng_states, 0, block_sizes[j])
    i3 = _cuRand(rng_states, 0, block_sizes[j])

    (p1, p2, p3) = (
        points[block_start_indices[j] + i1],
        points[block_start_indices[j] + i2],
        points[block_start_indices[j] + i3],
    )
    # w = plane_coefficients[j, i]  # cuda.local.array(shape=4, dtype=nb.float32)
    w = cuda.local.array(shape=4, dtype=nb.float32)
    ux, uy, uz = _cu_subtract_point(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
    vx, vy, vz = _cu_subtract_point(p2[0], p2[1], p2[2], p3[0], p3[1], p3[2])
    w[0], w[1], w[2] = _crossNormal(ux, uy, uz, vx, vy, vz)
    w[3] = -_cu_dot(w[0], w[1], w[2], p1[0], p1[1], p1[2])

    cc = 0.0
    sum = 0
    for jj in range(block_sizes[j]):
        p = points[block_start_indices[j] + jj]
        distance = math.fabs(_cu_dot(w[0], w[1], w[2], p[0], p[1], p[2]) + w[3])
        sum = sum + distance
        if distance < threshold:
            result_mask[i][block_start_indices[j] + jj] = 1


class CudaRansac:
    def __init__(
        self,
        threshold: float = 0.1,
        initial_points: int = 3,
        iterations: int = 5000,
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

    def fit(
        self,
        point_cloud: PointCloud,
        block_sizes: npt.NDArray,
        block_start_indices: npt.NDArray,
        n_blocks: int,
        n_threads_per_block: int,
    ):
        result_mask = np.zeros((n_threads_per_block, len(point_cloud)), dtype=np.int32)
        result_mask_cuda = cuda.to_device(result_mask)
        point_cloud_cuda = cuda.to_device(point_cloud)
        block_sizes_cuda = cuda.to_device(block_sizes)
        block_start_indices_cuda = cuda.to_device(block_start_indices)

        rng_states = create_xoroshiro128p_states(n_threads_per_block * n_blocks, seed=0)
        _do_fit[n_blocks, n_threads_per_block](
            point_cloud_cuda,
            block_sizes_cuda,
            block_start_indices_cuda,
            self.__threshold,
            result_mask_cuda,
            rng_states,
        )
        result_mask = result_mask_cuda.copy_to_host()
        maximum_mask = result_mask[np.argmax(result_mask.sum(axis=1), axis=0)]
        return maximum_mask
