import numpy as np
import numpy.typing as npt
import numba as nb
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from octreelib.internal import PointCloud
from octreelib.ransac.initial_points_config import N_INITIAL_POINTS
from octreelib.ransac.util import (
    generate_random_indices,
    get_plane_from_points,
    measure_distance,
)

__all__ = ["CudaRansac"]


N_CUDA_THREADS = 1024


@cuda.jit
def ransac_kernel(
    point_cloud: PointCloud,
    block_sizes: npt.NDArray,
    block_start_indices: npt.NDArray,
    threshold: float,
    rng_states,
    result_mask: npt.NDArray,
    max_n_inliers: npt.NDArray,
    mask_mutex: npt.NDArray,
):
    thread_id, block_id = cuda.threadIdx.x, cuda.blockIdx.x

    if block_sizes[block_id] < N_INITIAL_POINTS:
        return

    # select random points as inliers
    initial_point_indices = cuda.local.array(shape=N_INITIAL_POINTS, dtype=nb.size_t)
    initial_point_indices = generate_random_indices(
        initial_point_indices, rng_states, block_sizes[block_id], N_INITIAL_POINTS
    )
    for i in range(N_INITIAL_POINTS):
        initial_point_indices[i] = (
            block_start_indices[block_id] + initial_point_indices[i]
        )

    # calculate the plane coefficients
    plane = cuda.local.array(shape=4, dtype=nb.float32)
    plane[0], plane[1], plane[2], plane[3] = get_plane_from_points(
        point_cloud, initial_point_indices
    )

    # for each point in the block check if it is an inlier
    n_inliers_local = 0
    for i in range(block_sizes[block_id]):
        point = point_cloud[block_start_indices[block_id] + i]
        distance = measure_distance(plane, point)
        if distance < threshold:
            n_inliers_local += 1

    # replace the maximum number of inliers if the current number is greater
    cuda.atomic.max(max_n_inliers, block_id, n_inliers_local)
    cuda.syncthreads()
    # set the best mask index for this block
    # if this thread has the maximum number of inliers
    if (
        n_inliers_local == max_n_inliers[block_id]
        and cuda.atomic.cas(mask_mutex, block_id, 0, 1) == 0
    ):
        for i in range(block_sizes[block_id]):
            if (
                measure_distance(plane, point_cloud[block_start_indices[block_id] + i])
                < threshold
            ):
                result_mask[block_start_indices[block_id] + i] = True


class CudaRansac:
    """
    RANSAC implementation using CUDA.
    """

    def __init__(
        self,
        max_n_blocks: int,
        threshold: float = 0.01,
        n_hypotheses: int = N_CUDA_THREADS,
    ):
        """
        Initialize the RANSAC parameters.
        :param threshold: Distance threshold.
        :param n_hypotheses: Number of RANSAC hypotheses. (<= 1024)
        :param max_n_blocks: Maximum number of blocks.
        """

        self.__threshold: float = threshold
        self.__n_threads_per_block = min(n_hypotheses, N_CUDA_THREADS)
        # create random number generator states
        self.__rng_states = create_xoroshiro128p_states(
            self.__n_threads_per_block * max_n_blocks, seed=0
        )

    def evaluate(
        self,
        point_cloud: PointCloud,
        block_sizes: npt.NDArray,
    ):
        """
        Evaluate the RANSAC model.
        :param point_cloud: Point cloud to fit the model to.
        :param block_sizes: Array of block sizes (should equal number of leaf voxels).
        """

        n_blocks = len(block_sizes)

        # create result mask and copy it to the device
        result_mask_cuda = cuda.to_device(np.zeros((len(point_cloud)), dtype=np.bool_))

        # create arrays to store the maximum number of inliers and the best mask indices
        max_n_inliers_cuda = cuda.to_device(np.zeros(n_blocks, dtype=np.int32))

        # copy point_cloud, block_sizes and block_start_indices to the device
        point_cloud_cuda = cuda.to_device(point_cloud)
        block_sizes_cuda = cuda.to_device(block_sizes)
        # block_start_indices is an array of indices where each cuda block should
        # take data from this combined with block_sizes allows it to quickly
        # find the desired part of the point cloud
        block_start_indices_cuda = cuda.to_device(
            np.cumsum(np.concatenate(([0], block_sizes[:-1])))
        )

        # this mutex is needed to make sure that only one thread writes to the mask
        mask_mutex = cuda.to_device(np.zeros(n_blocks, dtype=np.int32))

        # call the kernel
        ransac_kernel[n_blocks, self.__n_threads_per_block](
            point_cloud_cuda,
            block_sizes_cuda,
            block_start_indices_cuda,
            self.__threshold,
            self.__rng_states,
            result_mask_cuda,
            max_n_inliers_cuda,
            mask_mutex,
        )

        # copy result mask back to the host
        result_mask = result_mask_cuda.copy_to_host()
        return result_mask
