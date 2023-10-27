from typing import Annotated, Literal, Iterable

import numpy as np
import numpy.typing as npt


__all__ = [
    "RawPoint",
    "RawPointCloud",
    "Point",
    "PointCloud",
    "PosePoint",
    "PosePointCloud",
]

# RawPoint is a numpy Array 1x3.
RawPoint = Annotated[npt.NDArray[np.float_], Literal[3]]
# RawPointCloud is a numpy array Nx3.
RawPointCloud = Annotated[npt.NDArray[np.float_], Literal["N", 3]]


# These classes are for internal use.
# They are subclasses of np.ndarray, which add dichotomy
# for having and not having information about pose number.

class Point(np.ndarray):
    def __new__(cls, input_array):
        # Construct from existing np.ndarray.
        obj = np.asarray(input_array).view(cls)
        return obj

    def with_pose(self, pose_number: int):
        # Return the same point but with information about pose numbers.
        new_array = np.hstack([self, np.array([pose_number])])
        return PosePoint(new_array)


class PointCloud(np.ndarray):
    def __new__(cls, input_array):
        # Construct from existing np.ndarray.
        obj = np.asarray(input_array).view(cls)
        return obj

    def with_poses(self, pose_numbers: Iterable[int]):
        # Return the same point cloud but with information about pose numbers.
        new_array = np.hstack([self, np.array(pose_numbers).reshape((len(self), 1))])
        return PosePointCloud(new_array)

    def with_pose(self, pose_numbers: int):
        # Return the same point cloud but with information about pose numbers.
        return self.with_poses([pose_numbers] * len(self))

    def __iter__(self) -> Point:
        # when iterating, return a Point class instead of a np.ndarray.
        for value in super().__iter__():
            yield Point(value)

    def __getitem__(self, index) -> Point:
        # getting item by index, return a Point class instead of a np.ndarray.
        return Point(super().__getitem__(index))

    def extend(self, other):
        # Add two point clouds.
        # Cannot override __add__ because it is used for different
        # purposes in parent np.ndarray.
        return PointCloud(np.vstack((self, other)))


class PosePoint(np.ndarray):
    def __new__(cls, input_array):
        # Construct from existing np.ndarray.
        obj = np.asarray(input_array).view(cls)
        return obj

    def without_pose(self):
        # Return the same point but without information about pose numbers.
        return Point(self[:3])

    def pose(self):
        # Return pose number.
        return self[3]


class PosePointCloud(np.ndarray):
    def __new__(cls, input_array):
        # Construct from existing np.ndarray.
        obj = np.asarray(input_array).view(cls)
        return obj

    def without_poses(self):
        # Return the same point cloud but without information about pose numbers.
        return PointCloud(self[:, :3])

    def poses(self):
        # Return poses for this cloud.
        return self[:, 3]

    def __iter__(self):
        # When iterating, return a Point class instead of a np.ndarray.
        for value in super().__iter__():
            yield PosePoint(value)

    def __getitem__(self, index):
        # When getting item by index, return a Point class instead of a np.ndarray.
        return PosePoint(super().__getitem__(index))

    def copy(self):
        # When copying return PosePointCloud instead of np.ndarray.
        return PosePointCloud(super().copy())

    def extend(self, other):
        # Add two point clouds.
        # Cannot override __add__ because it is used for different
        # purposes in parent np.ndarray.
        return PosePointCloud(np.vstack((self, other)))
