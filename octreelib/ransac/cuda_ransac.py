import numpy as np
import numpy.typing as npt
import numba as nb
from numba import cuda

from octreelib.internal import PointCloud
from octreelib.ransac.util import (
    get_plane_from_points,
    measure_distance,
)

__all__ = ["CudaRansac"]


CUDA_THREADS = 1024


class CudaRansac:
    """
    RANSAC implementation using CUDA.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        hypotheses_number: int = CUDA_THREADS,
        initial_points_number: int = 6,
    ):
        """
        Initialize the RANSAC parameters.
        :param threshold: Distance threshold.
        :param hypotheses_number: Number of RANSAC hypotheses. (<= 1024)
        :param initial_points_number: Number of initial points to use in RANSAC.
        """

        self.__threshold: float = threshold
        self.__threads_per_block = min(hypotheses_number, CUDA_THREADS)
        self.__kernel = self.__get_kernel(initial_points_number)
        self.__random_hypotheses_cuda = cuda.to_device(
            np.random.random((self.__threads_per_block, initial_points_number))
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

        # copy point_cloud, block_sizes and block_start_indices to the device
        point_cloud_cuda = cuda.to_device(point_cloud)
        block_sizes_cuda = cuda.to_device(block_sizes)
        # block_start_indices is an array of indices where each cuda block should
        # take data from this combined with block_sizes allows it to quickly
        # find the desired part of the point cloud
        block_start_indices_cuda = cuda.to_device(
            np.cumsum(np.concatenate(([0], block_sizes[:-1])))
        )

        planes_cuda = cuda.to_device(
            np.zeros((blocks_number, self.__threads_per_block, 4), dtype=np.float32)
        )

        # call the kernel
        self.__kernel[blocks_number, self.__threads_per_block](
            point_cloud_cuda,
            block_sizes_cuda,
            block_start_indices_cuda,
            self.__random_hypotheses_cuda,
            planes_cuda,
        )
        self.__find_best_plane[blocks_number, self.__threads_per_block](
            point_cloud_cuda,
            block_sizes_cuda,
            block_start_indices_cuda,
            self.__threshold,
            planes_cuda,
        )
        self.__compute_final_mask[blocks_number, self.__threads_per_block](
            point_cloud_cuda,
            block_sizes_cuda,
            block_start_indices_cuda,
            planes_cuda,
            self.__threshold,
            result_mask_cuda,
        )

        # copy result mask back to the host
        result_mask = result_mask_cuda.copy_to_host()
        return result_mask

    @staticmethod
    @cuda.jit
    def __find_best_plane(
        point_cloud: PointCloud,
        block_sizes: npt.NDArray,
        block_start_indices: npt.NDArray,
        threshold: float,
        planes: npt.NDArray,
    ):
        thread_id, block_id = cuda.threadIdx.x, cuda.blockIdx.x

        # for each point in the block check if it is an inlier
        inliers_number_local = 0
        for i in range(block_sizes[block_id]):
            point = point_cloud[block_start_indices[block_id] + i]
            distance = measure_distance(planes[block_id][thread_id], point)
            if distance < threshold:
                inliers_number_local += 1

        # shared memory to store the best plane and the maximum number of inliers
        # for all hypotheses
        max_inliers_number = cuda.shared.array(shape=1, dtype=nb.int32)
        # this mutex is needed to make sure that only one thread writes the best plane
        mutex = cuda.shared.array(shape=1, dtype=nb.int32)
        if thread_id == 0:
            max_inliers_number[0] = 0
            mutex[0] = 0
        cuda.syncthreads()

        # replace the maximum number of inliers if the current number is greater
        cuda.atomic.max(max_inliers_number, 0, inliers_number_local)

        # if this thread has the maximum number of inliers
        # write this thread's plane to the shared memory
        cuda.syncthreads()
        if (
            inliers_number_local == max_inliers_number[0]
            and cuda.atomic.compare_and_swap(mutex, 0, 1) == 0
        ):
            for i in range(4):
                planes[block_id][0][i] = planes[block_id][thread_id][i]
        cuda.syncthreads()

    @staticmethod
    @cuda.jit
    def __compute_final_mask(
        point_cloud: PointCloud,
        block_sizes: npt.NDArray,
        block_start_indices: npt.NDArray,
        planes: npt.NDArray,
        threshold: float,
        result_mask: npt.NDArray,
    ):
        thread_id, block_id = cuda.threadIdx.x, cuda.blockIdx.x
        for i in range(
            block_start_indices[block_id] + thread_id,
            block_start_indices[block_id] + block_sizes[block_id],
            cuda.blockDim.x,
        ):
            if measure_distance(planes[block_id][0], point_cloud[i]) < threshold:
                result_mask[i] = True

    @staticmethod
    def __get_kernel(initial_points_number):
        @cuda.jit
        def kernel(
            point_cloud: PointCloud,
            block_sizes: npt.NDArray,
            block_start_indices: npt.NDArray,
            random_hypotheses: npt.NDArray,
            planes: npt.NDArray,
        ):
            thread_id, block_id = cuda.threadIdx.x, cuda.blockIdx.x

            if block_sizes[block_id] < initial_points_number:
                return

            # select random points as inliers
            initial_point_indices = cuda.local.array(
                shape=initial_points_number, dtype=nb.size_t
            )
            for i in range(initial_points_number):
                initial_point_indices[i] = nb.int32(
                    random_hypotheses[thread_id][i] * block_sizes[block_id]
                    + block_start_indices[block_id]
                )

            # calculate the plane coefficients
            (
                planes[block_id][thread_id][0],
                planes[block_id][thread_id][1],
                planes[block_id][thread_id][2],
                planes[block_id][thread_id][3],
            ) = get_plane_from_points(point_cloud, initial_point_indices)

        return kernel
