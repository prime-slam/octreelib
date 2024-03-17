import numpy as np
import numpy.typing as npt
import numba as nb
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from octreelib.internal import PointCloud
from octreelib.ransac.util import generate_random_indices, measure_distance


N_INITIAL_POINTS = 6


@cuda.jit(device=True, inline=True)
def get_plane_from_points(points, initial_point_indices):
    """
    Calculate the plane coefficients from the given points.
    :param points: Point cloud.
    :param initial_point_indices: Inliers to calculate the plane coefficients from.
    """
    # This implementation works the same way as open3d implementation
    # ! but with some tweaks to make it work with numba and cuda !

    centroid_x, centroid_y, centroid_z = 0.0, 0.0, 0.0

    for idx in initial_point_indices:
        centroid_x += points[idx][0]
        centroid_y += points[idx][1]
        centroid_z += points[idx][2]

    centroid_x /= N_INITIAL_POINTS
    centroid_y /= N_INITIAL_POINTS
    centroid_z /= N_INITIAL_POINTS

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


@cuda.jit
def ransac_kernel(
    point_cloud: PointCloud,
    block_sizes: npt.NDArray,
    block_start_indices: npt.NDArray,
    threshold: float,
    result_mask: npt.NDArray,
    rng_states,
    max_n_inliers: npt.NDArray,
    mask_mutex: npt.NDArray,
):
    thread_id, block_id = cuda.threadIdx.x, cuda.blockIdx.x

    if block_sizes[block_id] < N_INITIAL_POINTS:
        return

    plane = cuda.local.array(shape=4, dtype=nb.float32)

    initial_point_indices = cuda.local.array(shape=N_INITIAL_POINTS, dtype=nb.size_t)
    initial_point_indices = generate_random_indices(
        initial_point_indices, rng_states, block_sizes[block_id], N_INITIAL_POINTS
    )
    for i in range(N_INITIAL_POINTS):
        initial_point_indices[i] = (
            block_start_indices[block_id] + initial_point_indices[i]
        )

    # calculate the plane coefficients
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
        threshold: float = 0.01,
        iterations: int = 1024,
        max_n_blocks: int = 1,
        n_threads_per_block: int = 1024,
    ):
        """
        Initialize the RANSAC parameters.
        :param threshold: Distance threshold.
        :param iterations: Number of RANSAC iterations. (has no effect in this implementation)
        :param max_n_blocks: Maximum number of blocks.
        :param n_threads_per_block: Number of threads per block.
        """
        # in this implementation the parameters are set in the constructor
        # the alternative would be to set them in the fit method
        if threshold <= 0:
            raise ValueError("Threshold must be positive")
        if iterations < 1:
            raise ValueError("Number of RANSAC iterations must be positive")

        self.__threshold: float = threshold
        self.__iterations: int = iterations
        # create random number generator states
        self.rng_states = create_xoroshiro128p_states(
            n_threads_per_block * max_n_blocks, seed=0
        )
        self.n_threads_per_block = n_threads_per_block

    def evaluate(
        self,
        point_cloud: PointCloud,
        block_sizes: npt.NDArray,
        block_start_indices: npt.NDArray,
    ):
        """
        Evaluate the RANSAC model.
        :param point_cloud: Point cloud to fit the model to.
        :param block_sizes: Array of block sizes (should equal number of leaf voxels).
        :param block_start_indices: Array of block start indices (leaf voxel separators).
        """

        n_blocks = len(block_sizes)

        # create result mask and copy it to the device
        result_mask_cuda = cuda.to_device(np.zeros((len(point_cloud)), dtype=np.bool_))

        # create arrays to store the maximum number of inliers and the best mask indices
        max_n_inliers_cuda = cuda.to_device(np.zeros(n_blocks, dtype=np.int32))

        # copy point_cloud, block_sizes and block_start_indices to the device
        point_cloud_cuda = cuda.to_device(point_cloud)
        block_sizes_cuda = cuda.to_device(block_sizes)
        block_start_indices_cuda = cuda.to_device(block_start_indices)

        # this mutex is needed to make sure that only one thread writes to the mask
        mask_mutex = cuda.to_device(np.zeros(n_blocks, dtype=np.int32))

        # call the kernel
        ransac_kernel[n_blocks, self.n_threads_per_block](
            point_cloud_cuda,
            block_sizes_cuda,
            block_start_indices_cuda,
            self.__threshold,
            result_mask_cuda,
            self.rng_states,
            max_n_inliers_cuda,
            mask_mutex,
        )

        # copy result mask back to the host
        result_mask = result_mask_cuda.copy_to_host()
        return result_mask
