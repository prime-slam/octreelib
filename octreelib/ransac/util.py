import math
import numba as nb

from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32


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
    return (nb.int32)(x * (upper_bound - lower_bound) + lower_bound)


@cuda.jit(device=True, inline=True)
def generate_random_indices(initial_point_indices, rng_states, block_size, n_points):
    """
    Generate random points from the given block.
    :param initial_point_indices: Array to store the initial point indices.
    :param rng_states: Random number generator states.
    :param block_size: Size of the block.
    :param n_points: Number of points to generate.
    """

    for i in range(n_points):
        initial_point_indices[i] = generate_random_int(rng_states, 0, block_size)
    return initial_point_indices


@cuda.jit(device=True, inline=True)
def generate_unique_random_indices(
    initial_point_indices, rng_states, block_size, n_points
):
    """
    Generate unique random points from the given block.
    :param initial_point_indices: Array to store the initial point indices.
    :param rng_states: Random number generator states.
    :param block_size: Size of the block.
    :param n_points: Number of points to generate.
    """
    for ii in range(n_points):
        initial_point_indices[ii] = generate_random_int(rng_states, 0, block_size)
        unique = False
        while not unique:
            unique = True
            for jj in range(ii):
                if initial_point_indices[ii] == initial_point_indices[jj]:
                    unique = False
            if not unique:
                initial_point_indices[ii] = (initial_point_indices[ii] + 1) % block_size
    return initial_point_indices
