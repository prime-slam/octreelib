import numpy as np
import numpy.typing as npt
import numba as nb
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from octreelib.internal import PointCloud
from octreelib.ransac.util import (
    generate_random_indices,
    get_plane_from_points,
    measure_distance,
    INITIAL_POINTS_NUMBER,
)

__all__ = ["CudaRansac"]


CUDA_THREADS = 1024


class CudaRansac:
    """
    RANSAC implementation using CUDA.
    """

    def __init__(
        self,
        max_blocks_number: int,
        threshold: float = 0.01,
        hypotheses_number: int = CUDA_THREADS,
    ):
        """
        Initialize the RANSAC parameters.
        :param threshold: Distance threshold.
        :param hypotheses_number: Number of RANSAC hypotheses. (<= 1024)
        :param max_blocks_number: Maximum number of blocks.
        """

        self.__threshold: float = threshold
        self.__threads_per_block = min(hypotheses_number, CUDA_THREADS)
        # create random number generator states
        self.__rng_states = create_xoroshiro128p_states(
            self.__threads_per_block * max_blocks_number, seed=0
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

        blocks_number = len(block_sizes)

        # create result mask and copy it to the device
        result_mask_cuda = cuda.to_device(np.zeros((len(point_cloud)), dtype=np.bool_))

        # create arrays to store the maximum number of inliers and the best mask indices
        max_inliers_number_cuda = cuda.to_device(
            np.zeros(blocks_number, dtype=np.int32)
        )

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
        mask_mutex = cuda.to_device(np.zeros(blocks_number, dtype=np.int32))

        # call the kernel
        self.__ransac_kernel[blocks_number, self.__threads_per_block](
            point_cloud_cuda,
            block_sizes_cuda,
            block_start_indices_cuda,
            self.__threshold,
            self.__rng_states,
            result_mask_cuda,
            max_inliers_number_cuda,
            mask_mutex,
        )

        # copy result mask back to the host
        result_mask = result_mask_cuda.copy_to_host()
        return result_mask

    @staticmethod
    @cuda.jit
    def __ransac_kernel(
        point_cloud: PointCloud,
        block_sizes: npt.NDArray,
        block_start_indices: npt.NDArray,
        threshold: float,
        rng_states,
        result_mask: npt.NDArray,
        max_inliers_number: npt.NDArray,
        mask_mutex: npt.NDArray,
    ):
        thread_id, block_id = cuda.threadIdx.x, cuda.blockIdx.x

        if block_sizes[block_id] < INITIAL_POINTS_NUMBER:
            return

        # select random points as inliers
        initial_point_indices = cuda.local.array(
            shape=INITIAL_POINTS_NUMBER, dtype=nb.size_t
        )
        initial_point_indices = generate_random_indices(
            initial_point_indices,
            rng_states,
            block_sizes[block_id],
            INITIAL_POINTS_NUMBER,
        )
        for i in range(INITIAL_POINTS_NUMBER):
            initial_point_indices[i] = (
                block_start_indices[block_id] + initial_point_indices[i]
            )

        # calculate the plane coefficients
        plane = cuda.local.array(shape=4, dtype=nb.float32)
        plane[0], plane[1], plane[2], plane[3] = get_plane_from_points(
            point_cloud, initial_point_indices
        )

        # for each point in the block check if it is an inlier
        inliers_number_local = 0
        for i in range(block_sizes[block_id]):
            point = point_cloud[block_start_indices[block_id] + i]
            distance = measure_distance(plane, point)
            if distance < threshold:
                inliers_number_local += 1

        # replace the maximum number of inliers if the current number is greater
        cuda.atomic.max(max_inliers_number, block_id, inliers_number_local)
        cuda.syncthreads()
        # set the best mask index for this block
        # if this thread has the maximum number of inliers
        if (
            inliers_number_local == max_inliers_number[block_id]
            and cuda.atomic.cas(mask_mutex, block_id, 0, 1) == 0
        ):
            for i in range(block_sizes[block_id]):
                if (
                    measure_distance(
                        plane, point_cloud[block_start_indices[block_id] + i]
                    )
                    < threshold
                ):
                    result_mask[block_start_indices[block_id] + i] = True
